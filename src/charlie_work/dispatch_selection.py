"""Dispatch- and review-dispatch candidate selection (issue #1283 Phase A).

Extracted verbatim from ``workflow.py``: the free-function family that
selects which issues to dispatch (or redispatch as rework) and which PRs to
dispatch reviewers for, plus the two frozen dataclasses this family returns.
``workflow.py`` re-exports every symbol here via a facade import block
(mirroring ``config.py``'s ``RunnerAllocationConfig`` re-export pattern), so
existing import paths and monkeypatch targets keep working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from .config import ReviewDispatchConfig
from .process_utils import is_pid_alive
from .state import (
    _REVIEW_DEAD_CLAIM_BACKSTOP_TIMEOUT_MINUTES,
    _REVIEW_STALE_CLAIM_TIMEOUT_MINUTES,
    is_claim_stale,
    load_state_locked,
)
from .worker import iter_workers


@dataclass(frozen=True)
class LocalReviewCapResult:
    """Result of applying the local-only review process cap (issue #370).

    Unlike ``ConcurrencyGovernorResult``, this is not a provider-rate-limit
    clamp: it only bounds the number of concurrent local Claude Code reviewer
    processes, mirroring ``max_local_review_processes`` in
    ``ReviewDispatchConfig``. A value of 0 means unlimited.
    """

    clamped: bool
    max_local: int
    live_count: int
    available_slots: int
    dispatch_limit: int

    def report_fields(self) -> dict[str, int]:
        """Return the fields to include in CommandResult.data."""
        return {
            "max_local_review_processes": self.max_local,
            "live_review_count": self.live_count,
            "available_review_slots": self.available_slots,
        }


@dataclass(frozen=True)
class ReviewDispatchSelection:
    """Read-only result of selecting review-dispatch candidates (issue #617).

    Shared by the dry-run and real ``dispatch_reviews`` branches so the two
    cannot diverge. The real branch additionally runs attempt-cap escalation
    (a state mutation) separately; at-cap PRs are filtered by
    ``_is_review_dispatchable`` regardless of escalation status, so the
    ``dispatchable`` list is stable across that mutation.
    """

    escalated_skipped: list[int]
    merge_conflict_routed: list[dict[str, Any]]
    dispatchable: list[dict[str, Any]]
    local_cap: LocalReviewCapResult
    dispatch_limit: int
    selected: list[dict[str, Any]]


def parse_issue_numbers(only_issues: str) -> list[int]:
    return [int(part) for part in only_issues.replace(" ", "").split(",") if part]


# Maximum recovery-retry candidates allowed per dispatch pass. Recovery retries
# must not consume the same budget as fresh candidates; capping them at one per
# pass prevents one stuck recovery candidate from starving the queue (issue #506).
_MAX_RECOVERY_RETRY_PER_PASS = 1

# Maximum example issue numbers carried in the persisted deferred-by-concurrency
# field. A standing clamp (governor at 0 for hours) would otherwise re-emit the
# full candidate list every pass into events.db -- the same per-pass repetition
# dispatch_skip_blocked had to grow a dedup for. Follows the
# backlog_reachability.unreachable_examples idiom: a full count alongside a
# truncated example list (issue #1005).
_MAX_DEFERRED_CONCURRENCY_EXAMPLES = 5


def _is_recovery_candidate(
    issue: dict[str, Any],
    state: dict[str, Any],
    branch_name_for: Callable[[dict[str, Any]], str],
) -> bool:
    """Return True when ``issue`` is a recovery retry of a previous dispatch.

    A recovery candidate has a state.json entry whose status is ``"dispatched"``
    and whose stored branch_name matches the branch that would be generated for
    the issue today. These candidates are separated from fresh dispatch
    candidates so they cannot monopolize the dispatch budget.
    """
    issue_number = int(issue["number"])
    prev_entry = state.get("issues", {}).get(str(issue_number), {})
    if prev_entry.get("status") != "dispatched":
        return False
    prev_branch = prev_entry.get("branch_name")
    return prev_branch == branch_name_for(issue)


def _select_dispatch_candidates(
    candidates: list[dict[str, Any]],
    dispatch_limit: int,
    state: dict[str, Any],
    branch_name_for: Callable[[dict[str, Any]], str],
    only_issues: str | None = None,
) -> tuple[list[dict[str, Any]], list[int], list[int], int]:
    """Select dispatch candidates, filling fresh slots before recovery retries.

    Fresh candidates are dispatched first; recovery-retry candidates are only
    attempted with remaining slots, and at most one recovery retry is attempted
    per pass. This prevents a stuck recovery candidate from head-of-line
    blocking fresh work under a tight dispatch limit (issue #506).

    Args:
        candidates: Unblocked, sorted candidate issues from GitHub.
        dispatch_limit: Maximum number of issues to select this pass.
        state: Current state.json snapshot for recovery classification.
        branch_name_for: Callable that returns the branch name for an issue.
        only_issues: Optional explicit comma-separated issue numbers to select.

    Returns:
        Tuple of (selected, skipped_issue_numbers, deferred_by_concurrency,
        deferred_by_concurrency_count). ``deferred_by_concurrency`` is the
        FULL, untruncated list of every ordered candidate that was not
        selected -- populated on both the ``only_issues`` path and the
        automatic path (issue #1005; the automatic path used to report this
        unconditionally as ``[]``, making a saturated governor
        indistinguishable from an empty backlog).
        ``deferred_by_concurrency_count`` is ``len(deferred_by_concurrency)``,
        returned explicitly so callers don't have to re-derive it.

        Callers that persist this into a durable payload (a ``dispatch``
        event or ``CommandResult.data``) MUST truncate it themselves to
        ``_MAX_DEFERRED_CONCURRENCY_EXAMPLES`` entries before writing it --
        otherwise a standing clamp re-emits the full candidate list every
        pass. But the FULL list must still reach ``_build_failure_map``: a
        prior version of this fix truncated before returning, which silently
        dropped ``failures`` map entries for the 6th+ deferred issue on the
        ``only_issues`` path (a regression vs. pre-#1005 behavior, caught in
        review). Truncate at the payload call site, never inside this
        function.
    """
    if only_issues:
        wanted = parse_issue_numbers(only_issues)
        by_number = {int(issue["number"]): issue for issue in candidates}
        ordered = [by_number[number] for number in wanted if number in by_number]
        skipped_issue_numbers = sorted(set(wanted) - set(by_number))
    else:
        ordered = candidates
        skipped_issue_numbers = []

    recovery_flags = [_is_recovery_candidate(issue, state, branch_name_for) for issue in ordered]
    fresh_count = sum(1 for r in recovery_flags if not r)
    recovery_cap = min(
        _MAX_RECOVERY_RETRY_PER_PASS,
        max(0, dispatch_limit - fresh_count),
    )

    selected: list[dict[str, Any]] = []
    recovery_picked = 0
    if only_issues:
        # Preserve the operator's explicit issue order while still capping
        # recovery retries so a stuck recovery issue cannot starve fresh work.
        for issue, is_recovery in zip(ordered, recovery_flags):
            if len(selected) >= dispatch_limit:
                break
            if is_recovery:
                if recovery_picked < recovery_cap:
                    selected.append(issue)
                    recovery_picked += 1
            else:
                selected.append(issue)
    else:
        fresh = [issue for issue, r in zip(ordered, recovery_flags) if not r]
        recovery = [issue for issue, r in zip(ordered, recovery_flags) if r]
        # Fill fresh first, then allow at most one recovery-retry slot.
        selected = fresh[:dispatch_limit] + recovery[:recovery_cap]

    # Every ordered candidate not selected was deferred by the concurrency cap
    # -- true on both paths. (Candidates absent from ``ordered`` entirely, e.g.
    # an --issues number GitHub never returned, are already counted in
    # ``skipped_issue_numbers``, not here.)
    selected_numbers = {int(issue["number"]) for issue in selected}
    deferred_by_concurrency = [
        int(issue["number"]) for issue in ordered if int(issue["number"]) not in selected_numbers
    ]
    deferred_by_concurrency_count = len(deferred_by_concurrency)

    return selected, skipped_issue_numbers, deferred_by_concurrency, deferred_by_concurrency_count


def _select_rework_candidates(
    candidates: list[dict[str, Any]],
    rework_limit: int,
    only_issues: str | None = None,
) -> tuple[list[dict[str, Any]], list[int], list[int], int]:
    """Select rework dispatch candidates under the concurrency cap.

    Mirrors ``_select_dispatch_candidates`` for the fresh-dispatch path, but
    without the fresh-before-recovery ordering: every rework candidate is
    already in ``rework_requested`` state, so there is no fresh/recovery split
    to honor. Selection is a straight ``ordered[:rework_limit]`` cap.

    Args:
        candidates: Filtered rework candidate issues (each with an open PR).
        rework_limit: Maximum number of issues to select this pass (already
            concurrency-governor-clamped by the caller).
        only_issues: Optional explicit comma-separated issue numbers to select.

    Returns:
        Tuple of (selected, deferred_by_concurrency_full,
        deferred_by_concurrency, deferred_by_concurrency_count).

        ``deferred_by_concurrency_full`` is the FULL, untruncated list of every
        ordered candidate that was not selected -- populated on both the
        ``only_issues`` path and the automatic path (issue #1014, mirroring
        #1005 in the fresh-dispatch path; the automatic path used to report
        this unconditionally as ``[]``, making a saturated governor
        indistinguishable from an empty backlog). It MUST be fed to
        ``_build_failure_map`` so every deferred issue keeps its per-issue
        failures entry.

        ``deferred_by_concurrency`` is the same list truncated to
        ``_MAX_DEFERRED_CONCURRENCY_EXAMPLES`` entries for the persisted
        event / ``CommandResult.data`` field -- a standing clamp would
        otherwise re-emit the full candidate list every pass.

        ``deferred_by_concurrency_count`` is ``len(deferred_by_concurrency_full)``,
        returned explicitly so callers don't have to re-derive it.
    """
    if only_issues:
        wanted = parse_issue_numbers(only_issues)
        by_number = {int(issue["number"]): issue for issue in candidates}
        ordered = [by_number[number] for number in wanted if number in by_number]
    else:
        ordered = candidates
    selected = ordered[:rework_limit]
    selected_numbers = {int(issue["number"]) for issue in selected}
    deferred_by_concurrency_full = [
        int(issue["number"]) for issue in ordered if int(issue["number"]) not in selected_numbers
    ]
    deferred_by_concurrency_count = len(deferred_by_concurrency_full)
    deferred_by_concurrency = deferred_by_concurrency_full[:_MAX_DEFERRED_CONCURRENCY_EXAMPLES]
    return (
        selected,
        deferred_by_concurrency_full,
        deferred_by_concurrency,
        deferred_by_concurrency_count,
    )


def _reviewer_pid_alive(entry: dict[str, Any]) -> bool:
    """Check if a reviewer PID from state.json is alive, with start-time verification.

    Mirror of ``_worker_pid_alive`` for reviewer processes launched by
    ``dispatch_reviews``. A reviewer is a Claude Code worker whose PID and
    process start time are stored under ``reviewer_pid`` and
    ``reviewer_process_start_time`` in the per-PR state.
    """
    reviewer_pid = entry.get("reviewer_pid")
    if reviewer_pid is None:
        return False

    return is_pid_alive(reviewer_pid, entry.get("reviewer_process_start_time"))


def _count_live_reviews(reviews_dir: Path, state_file: Path | None = None) -> int:
    """Count the number of currently alive reviewer sessions.

    Reads review sidecar files (claude-code only today) from ``reviews_dir``,
    then checks each record's PID liveness using the adapter-agnostic
    ``WorkerView.is_alive`` probe.

    When ``state_file`` is given, the count is corroborated against
    state.json's own ``review_dispatch_dispatched`` records so a missing
    sidecar doesn't let a live reviewer slip through the local cap.
    """
    live_pr_numbers: set[int] = set()
    live_count = 0
    for w in iter_workers(reviews_dir):
        if w.is_alive():
            live_count += 1
            live_pr_numbers.add(w.issue_number)

    if state_file is not None:
        state = load_state_locked(state_file)
        for pr_number_str, entry in state.get("prs", {}).items():
            if not isinstance(entry, dict):
                continue
            if entry.get("review_dispatch_status") != "review_dispatch_dispatched":
                continue
            try:
                pr_number = int(pr_number_str)
            except (TypeError, ValueError):
                continue
            if pr_number in live_pr_numbers:
                continue
            if _reviewer_pid_alive(entry):
                live_count += 1
                print(
                    f"[reconcile] PR {pr_number}: reviewer_pid {entry.get('reviewer_pid')} "
                    "is alive with no live review sidecar (ghost) -- counting it against "
                    "the local review cap instead of treating the slot as free",
                    flush=True,
                )

    return live_count


def _apply_local_review_cap(
    dispatch_limit: int,
    max_local: int,
    live_count: int,
) -> LocalReviewCapResult:
    """Apply the local-only review process cap to a dispatch limit.

    ``max_local_review_processes`` is a per-host safety valve, not a provider
    governor. A value of 0 means unlimited. Unlike ``_apply_concurrency_governor``,
    this never consults fleet-wide concurrency or rate-limit state.
    """
    if max_local <= 0:
        return LocalReviewCapResult(
            clamped=False,
            max_local=0,
            live_count=live_count,
            available_slots=dispatch_limit,
            dispatch_limit=dispatch_limit,
        )

    available = max(0, max_local - live_count)
    clamped = available < dispatch_limit
    return LocalReviewCapResult(
        clamped=clamped,
        max_local=max_local,
        live_count=live_count,
        available_slots=available,
        dispatch_limit=min(dispatch_limit, available),
    )


def _windowed_redispatch_at(
    entry: dict[str, Any],
    *,
    window_minutes: int,
) -> list[str]:
    """Return redispatch timestamps within the configured window, type-safely.

    Normalizes ``entry["redispatch_at"]`` to a list of strings, filtering out
    non-string entries and timestamps older than ``window_minutes`` from now.
    This prevents crashes when the persisted value is corrupted (e.g., a string
    instead of a list — ``list("abc")`` would yield individual characters that
    crash ``datetime.fromisoformat``).
    """
    raw = entry.get("redispatch_at")
    if not isinstance(raw, list):
        return []
    now = datetime.now(UTC)
    window_start = now - timedelta(minutes=window_minutes)
    result: list[str] = []
    for t in raw:
        if not isinstance(t, str):
            continue
        try:
            if datetime.fromisoformat(t.replace("Z", "+00:00")) >= window_start:
                result.append(t)
        except (ValueError, AttributeError):
            continue
    return result


def _windowed_worker_death_at(
    entry: dict[str, Any],
    *,
    window_minutes: int,
) -> list[str]:
    """Return worker-death timestamps within the configured window, type-safely.

    Parallel to ``_windowed_redispatch_at`` but reads
    ``entry["worker_death_at"]`` — the list of timestamps recorded by the
    orphan sweep each time it recovers a dead rework worker whose PR head
    has not moved (issue #1134).  A death is not a no-op: the worker may
    have completed its work but died before pushing.  Counting deaths
    against the no-op rework cap mislabels salvageable stranded work as
    "worker produced nothing."  This helper lets the no-op cap check
    separate death redispatches from genuine no-op redispatches.
    """
    raw = entry.get("worker_death_at")
    if not isinstance(raw, list):
        return []
    now = datetime.now(UTC)
    window_start = now - timedelta(minutes=window_minutes)
    result: list[str] = []
    for t in raw:
        if not isinstance(t, str):
            continue
        try:
            if datetime.fromisoformat(t.replace("Z", "+00:00")) >= window_start:
                result.append(t)
        except (ValueError, AttributeError):
            continue
    return result


def _windowed_orphan_redispatch_at(
    entry: dict[str, Any],
    *,
    window_minutes: int,
) -> list[str]:
    """Return orphan-sweep redispatch timestamps within the configured window.

    Parallel to ``_windowed_redispatch_at`` and ``_windowed_worker_death_at``
    but reads ``entry["orphan_redispatch_at"]`` -- the list of timestamps
    recorded by the orphan-sweep no-open-PR redispatch cap (issue #1243) each
    time it processes an issue whose worker died without leaving an open PR.
    Unlike ``adapter_history`` (which only grew when ``api_worker.enabled``
    was ``True``, before the per-issue adapter selector was deleted in
    Phase 2 Track B), this list grows in
    the default (non-API-routed) configuration too. It is appended once per
    *dead dispatch* (keyed by ``orphan_redispatch_counted_dispatch``), not
    once per sweep pass -- the #417 reclaim leaves the dead-worker record in
    place, so the same entry is re-observed every pass until a genuine
    redispatch replaces it.
    """
    raw = entry.get("orphan_redispatch_at")
    if not isinstance(raw, list):
        return []
    now = datetime.now(UTC)
    window_start = now - timedelta(minutes=window_minutes)
    result: list[str] = []
    for t in raw:
        if not isinstance(t, str):
            continue
        try:
            if datetime.fromisoformat(t.replace("Z", "+00:00")) >= window_start:
                result.append(t)
        except (ValueError, AttributeError):
            continue
    return result


def _windowed_blocked_environment_at(
    entry: dict[str, Any],
    *,
    window_minutes: int,
) -> list[str]:
    """Return pre-launch environment-block timestamps within the window.

    Parallel to ``_windowed_redispatch_at`` but reads
    ``entry["blocked_environment_at"]`` -- the list of timestamps recorded
    each time a rework/fresh dispatch fails at launch with a
    ``PRE_LAUNCH_BLOCKED_ENVIRONMENT_FAILURE_KINDS`` failure_kind (issue
    #1393).  These are pre-launch environment conflicts (e.g. a stale
    foreign worktree) that never started a worker session, so they must
    NOT count against the redispatch cap (which measures worker output,
    not environment hygiene).  A separate counter lets the dispatch layer
    escalate with the correct reason (``dispatch_blocked_environment``)
    and the blocking path after the same ``max_auto_redispatch`` budget.
    """
    raw = entry.get("blocked_environment_at")
    if not isinstance(raw, list):
        return []
    now = datetime.now(UTC)
    window_start = now - timedelta(minutes=window_minutes)
    result: list[str] = []
    for t in raw:
        if not isinstance(t, str):
            continue
        try:
            if datetime.fromisoformat(t.replace("Z", "+00:00")) >= window_start:
                result.append(t)
        except (ValueError, AttributeError):
            continue
    return result


def _windowed_foreign_writer_reaps(
    entry: dict[str, Any],
    *,
    window_minutes: int,
) -> list[str]:
    """Return foreign-writer reap timestamps within the window.

    Issue #1423: parallel to ``_windowed_blocked_environment_at`` but reads
    ``entry["foreign_writer_reaps"]`` -- the list of timestamps recorded each
    time a ``worktree_foreign_writer`` block was auto-reaped at the
    blocked-environment cap exhaustion point (instead of escalating). Each
    successful reap resets ``blocked_environment_at`` to ``[]``, so without
    this separate cross-pass counter a persistently-blocked worktree would
    loop forever between reap and redispatch. The cap is enforced by the
    caller against ``watchdog.max_foreign_writer_reaps``; when the windowed
    count is at/over the cap, the caller escalates instead of reaping.
    """
    raw = entry.get("foreign_writer_reaps")
    if not isinstance(raw, list):
        return []
    now = datetime.now(UTC)
    window_start = now - timedelta(minutes=window_minutes)
    result: list[str] = []
    for t in raw:
        if not isinstance(t, str):
            continue
        try:
            if datetime.fromisoformat(t.replace("Z", "+00:00")) >= window_start:
                result.append(t)
        except (ValueError, AttributeError):
            continue
    return result


def _is_review_dispatchable(
    state: dict[str, Any],
    pr_number: int,
    candidate: dict[str, Any],
    *,
    max_attempts: int = 3,
    now: datetime | None = None,
    max_consecutive_turn_limit_misses: int = 0,
) -> bool:
    """Return True if ``pr_number`` is free to receive a new reviewer dispatch.

    A PR is dispatchable when:
    - No prior review dispatch claim exists.
    - A prior claim is terminal (completed or stale-failed) and the stale timeout
      has elapsed, allowing retry.
    - A dispatched reviewer is no longer alive and its claim has gone stale.
    - The per-PR dispatch attempt count has not reached ``max_attempts``.
    - The per-PR consecutive turn-limit miss streak has not reached
      ``max_consecutive_turn_limit_misses`` (issue #1439). 0 disables this
      backstop.

    This reuses ``is_claim_stale`` for the timeout and ``_reviewer_pid_alive``
    for liveness, avoiding a parallel mechanism.

    ``now`` is the injectable clock (issue #828), forwarded to every
    ``is_claim_stale`` call below: defaults to ``datetime.now(UTC)`` there
    when not supplied, so production behavior is byte-identical. Callers
    evaluating a whole candidate list against one shared instant (e.g.
    ``dispatch_reviews``) should sample ``now`` once and pass the same value
    to each ``_is_review_dispatchable`` call.
    """
    pr_state = state["prs"].get(str(pr_number), {})
    status = pr_state.get("review_dispatch_status")

    # Per-PR dispatch attempt cap: a PR that has been dispatched max_attempts
    # times without producing a verdict is stuck (e.g. every reviewer hits the
    # session limit). Escalation is handled by the caller; here we just block
    # further dispatch.
    attempt_count = int(pr_state.get("review_dispatch_attempt_count", 0))
    if attempt_count >= max_attempts:
        return False

    # Issue #1439: turn-limit miss backstop. A PR that has hit the turn limit
    # ``max_consecutive_turn_limit_misses`` times in a row (with the cap
    # already maxed at ``turn_cap_max_multiplier``) is not going to converge
    # on another identical session -- escalate instead of redispatching. The
    # caller escalates; here we just block further dispatch. 0 disables.
    if max_consecutive_turn_limit_misses > 0:
        miss_streak = int(pr_state.get("review_turn_limit_miss_streak", 0))
        if miss_streak >= max_consecutive_turn_limit_misses:
            return False

    if status is None or status == "review_dispatch_completed":
        return True

    if status == "review_dispatch_pending":
        pending_at = pr_state.get("review_dispatch_pending_at")
        return pending_at is None or is_claim_stale(
            pending_at, timeout_minutes=_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES, now=now
        )

    if status == "review_dispatch_dispatched":
        if _reviewer_pid_alive(pr_state):
            return False
        # Dead reviewer: the stalled-review sweep must disposition this claim
        # first — it classifies throttle tails, counts probe failures, reaps
        # the sidecar, and emits events; freeing here does none of that.
        # Racing the sweep on the same 5-minute timeout measured later in the
        # pass livelocked probes during closed quota windows (issue #571):
        # each silent relaunch reset the clock just after the sweep looked,
        # so backoff never engaged and a 429'd probe launched every pass.
        # The longer backstop frees only true orphans the sweep cannot see
        # (e.g. died before its sidecar was written).
        dispatched_at = pr_state.get("review_dispatched_at")
        return dispatched_at is None or is_claim_stale(
            dispatched_at, timeout_minutes=_REVIEW_DEAD_CLAIM_BACKSTOP_TIMEOUT_MINUTES, now=now
        )

    if status == "review_dispatch_failed":
        failed_at = pr_state.get("review_dispatch_failed_at")
        return failed_at is None or is_claim_stale(
            failed_at, timeout_minutes=_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES, now=now
        )

    # Unknown status: treat as free so we don't silently orphan PRs.
    return True


def _select_review_dispatch_candidates(
    candidates: list[dict[str, Any]],
    state: dict[str, Any],
    review_dispatch_config: ReviewDispatchConfig,
    reviews_dir: Path,
    state_file: Path,
    resolved_now: datetime,
    limit: int | None,
    probe_mode: bool,
) -> ReviewDispatchSelection:
    """Read-only selection of review-dispatch candidates (issue #617 rework).

    Single source of truth for the PR-selection logic shared by the dry-run
    and real ``dispatch_reviews`` branches, so the two cannot diverge:
    - escalated-skip (PRs/issues already at ``"escalated"`` status)
    - merge-conflict routing (CONFLICTING/DIRTY PRs, with live-reviewer
      protection so a mid-session reviewer is not routed to rework)
    - dispatchable-list construction (filtered by ``_is_review_dispatchable``)
    - local/concurrent cap computation and final selection

    Entirely read-only: never mutates ``state`` or writes to ``state_file``.
    The real branch's attempt-cap escalation (which mutates state) runs
    separately; at-cap PRs are filtered by ``_is_review_dispatchable``
    regardless of whether they have already been escalated, so the
    ``dispatchable`` list is stable across that mutation.

    ``candidates`` must already have rescue-marked PRs excluded — this
    function does not re-check ``rescue_attempted``. The dry-run branch
    excludes them via a read-only state check (not ``_partition_rescue_candidates``,
    which has real side effects); the real branch excludes them via
    ``_partition_rescue_candidates`` before calling this helper.
    """
    max_attempts = review_dispatch_config.max_review_dispatch_attempts
    max_turn_limit_misses = review_dispatch_config.max_consecutive_turn_limit_misses
    escalated_skipped: list[int] = []
    merge_conflict_routed: list[dict[str, Any]] = []
    for c in candidates:
        pr_state = state.get("prs", {}).get(str(c["pr"]), {})
        issue_num_gate = pr_state.get("issue_number") or c.get("issue")
        issue_state_gate = (
            state.get("issues", {}).get(str(issue_num_gate), {})
            if issue_num_gate is not None
            else {}
        )
        if pr_state.get("status") == "escalated" or issue_state_gate.get("status") == "escalated":
            escalated_skipped.append(c["pr"])
            continue
        if (
            str(c.get("mergeable") or "").upper() == "CONFLICTING"
            or str(c.get("mergeStateStatus") or "").upper() == "DIRTY"
        ):
            if pr_state.get(
                "review_dispatch_status"
            ) == "review_dispatch_dispatched" and _reviewer_pid_alive(pr_state):
                # Same live-reviewer protection as the attempt-cap path: do
                # not route to rework while a reviewer is mid-session.
                continue
            merge_conflict_routed.append(c)
            continue
    escalated_skipped_set = set(escalated_skipped)
    merge_conflict_pr_set = {c["pr"] for c in merge_conflict_routed}
    dispatchable = [
        c
        for c in candidates
        if c["pr"] not in escalated_skipped_set
        and c["pr"] not in merge_conflict_pr_set
        and _is_review_dispatchable(
            state,
            c["pr"],
            c,
            max_attempts=max_attempts,
            now=resolved_now,
            max_consecutive_turn_limit_misses=max_turn_limit_misses,
        )
    ]
    max_local = review_dispatch_config.max_local_review_processes
    max_concurrent = review_dispatch_config.max_concurrent_reviews
    live_count = _count_live_reviews(reviews_dir, state_file)
    requested_limit = limit if limit is not None else len(dispatchable)
    local_cap = _apply_local_review_cap(requested_limit, max_local, live_count)
    if max_concurrent > 0:
        concurrent_available = max(0, max_concurrent - live_count)
        concurrent_cap = min(local_cap.dispatch_limit, concurrent_available)
    else:
        concurrent_cap = local_cap.dispatch_limit
    dispatch_limit = 1 if probe_mode else concurrent_cap
    selected = dispatchable[:dispatch_limit]
    return ReviewDispatchSelection(
        escalated_skipped=escalated_skipped,
        merge_conflict_routed=merge_conflict_routed,
        dispatchable=dispatchable,
        local_cap=local_cap,
        dispatch_limit=dispatch_limit,
        selected=selected,
    )
