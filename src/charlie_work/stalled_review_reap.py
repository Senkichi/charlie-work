"""Stalled-review-reap free-function family (issue #1283 Phase A, PR 6 of 6).

Extracted verbatim from ``workflow.py``: the ten functions/class that detect
and reap stalled or orphaned code-review dispatch state --
``_remove_review_checkout_with_warning``, ``_set_reviewer_quota_exhausted_with_backoff``,
``_merge_on_write_save``, ``_ThrottleClassification``,
``_detect_and_handle_stalled_reviews``, ``_reap_review_sidecar``,
``_reap_completed_review_checkouts``, ``_reap_orphaned_review_checkouts``,
``_classify_review_dispatch_stalled_level``, and ``_append_sweep_events``.

This is a staged split of issue #1283's originally-scoped 35-member
dead-worker/stalled-review family: the 25-member dead-worker/session half is
spun off to issue #1317 (a deliberate refactor outside this split lane) and
ships separately; only the 10-member stalled-review-reap half moves here.

Corrected cohesion rationale (do not repeat an earlier "zero call-graph
edges, purely physical adjacency" framing for members 6/7 -- it is
factually wrong): ``_reap_completed_review_checkouts`` calls
``_reap_review_sidecar`` directly, and ``_reap_orphaned_review_checkouts``
also calls ``_reap_review_sidecar`` directly. All ten members are either
directly call-graph-connected to another member, or -- for
``_ThrottleClassification`` (a pure ``Enum``) and
``_classify_review_dispatch_stalled_level`` (called only by
``_detect_and_handle_stalled_reviews``) -- a tightly-scoped intra-group
dependency.

This module is expected to exceed the repo's normal 800-line-per-module
cap; that gate is explicitly waived for this extraction by operator
decision on issue #1283 (staged-split comment, 2026-08-17).

``workflow.py`` re-exports every symbol here via a facade import block
(mirroring ``config.py``'s ``RunnerAllocationConfig`` re-export pattern and
this repo's own ``dispatch_selection.py``/``escalation.py``/
``verdict_parsing.py``/``rework_prompts.py``/``ci_findings.py``/
``backlog_reachability.py`` precedents), so existing import paths and
monkeypatch targets keep working unchanged.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from .config import OrchestratorConfig
from .github import GitHubLike
from .instrumentation import log_event
from .process_utils import is_pid_alive
from .state import (
    _REVIEW_STALE_CLAIM_TIMEOUT_MINUTES,
    is_claim_stale,
    load_state,
    load_state_locked,
    reviewer_quota_last_probe_cleared_at,
    set_reviewer_quota_exhausted,
    state_lock,
    utc_now,
    without_review_dispatch_claim,
)
from .throttle_signatures import match_throttle_tail, parse_reset_clock_time
from .worker import _alive_review_worker_issue_numbers, iter_workers
from .worktree import remove_review_checkout
from .write_gate import WriteGate, require_write_gate


def _remove_review_checkout_with_warning(
    state: dict[str, Any],
    repo_root: Path,
    reviews_dir: Path,
    pr_number: int,
    write_gate: WriteGate,
    *,
    state_file: Path | None = None,
) -> tuple[dict[str, Any], bool]:
    """Remove an isolated review checkout and emit a one-shot warning on failure.

    Returns ``(new_state, removed)``. On failure, sets a per-PR warning marker
    in ``state["prs"][pr_number]`` and appends a ``review_checkout_removal_failed``
    event, but only if that PR does not already have an active warning marker.
    The marker is cleared when removal succeeds so a future failure can alert
    again. Retry happens on the next sweep pass; the warning is never re-emitted
    per pass.

    ``state_file`` is now vestigial at this call site: the sole write below
    goes through ``write_gate`` (which carries its own bound ``state_path``)
    rather than the raw ``append_event(state_path=state_file)`` this used to
    call directly. Kept rather than removed -- W6 PR2 (#1264) is a signature
    addition, not a signature cleanup, and every existing caller already
    passes it.
    """
    write_gate = require_write_gate(write_gate)
    removed = remove_review_checkout(repo_root, pr_number, reviews_dir=reviews_dir)
    pr_key = str(pr_number)
    prs = state.get("prs")
    if not isinstance(prs, dict):
        prs = {}
    pr_state = prs.get(pr_key)
    if not isinstance(pr_state, dict):
        pr_state = {}

    if removed:
        if pr_state.get("review_checkout_removal_warned"):
            state = {
                **state,
                "prs": {**prs, pr_key: {**pr_state, "review_checkout_removal_warned": None}},
            }
        return state, True

    if not pr_state.get("review_checkout_removal_warned"):
        state = {
            **state,
            "prs": {**prs, pr_key: {**pr_state, "review_checkout_removal_warned": True}},
        }
        state = write_gate.append_event(
            state,
            "review_checkout_removal_failed",
            {"pr_number": pr_number, "reviews_dir": str(reviews_dir)},
        )
    return state, False


def _set_reviewer_quota_exhausted_with_backoff(
    state: dict[str, Any],
    config: OrchestratorConfig,
    now_dt: datetime,
    *,
    reset_at: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Record a quota-exhaustion episode with exponential probe backoff.

    Every consecutive throttle hit without an intervening successful probe
    (a "successful probe" is a recorded verdict from a dead reviewer -- see
    dispatch_reviews's verdict-reap clear, the only proof the quota window is
    actually open) doubles the probe interval, capped at
    ``quota_probe_max_interval_minutes``, so a live provider outage does not
    relaunch a real reviewer session into the wall every
    ``quota_probe_interval_minutes`` forever (cost-spirals.md Finding 2: the
    config comment used to say "No escalation backoff" and meant it literally
    -- provider-throttle stalls are also exempt from the per-PR dispatch
    attempt cap, so this was the one failure mode that could not terminate).
    ``consecutive_probe_failures`` lives inside the existing ``reviewer_quota``
    dict rather than as a new state.py-owned field/helper, matching this
    fix's file scope.

    Issue #612: when the provider's session-limit notice names a specific
    reset time (``reset_at``, the parsed "resets H:MMam/pm (zone)" clock
    time), ``throttled_until`` is that reset plus the configured
    ``throttle_resume_margin_s`` instead of ``now + quota_reset_hours``.
    The provider's own stated reset is a far better backoff target than a
    fixed guess: it avoids both re-spending into a still-closed window
    (fixed window shorter than the real reset) and stalling far longer than
    necessary (fixed window longer than the real reset). The resume margin
    is added because provider reset estimates are floors, not guarantees
    (issue #499). When ``reset_at`` is None (no clock-time notice parsed,
    or the named zone was unavailable), the fixed ``quota_reset_hours``
    window is used as before.

    Returns ``(new_state, quota_record)`` where ``quota_record`` is the
    written ``reviewer_quota`` dict, so callers can emit a
    ``review_quota_exhausted`` event carrying ``throttled_until``,
    ``probe_after``, ``reset_at`` (ISO or None), and
    ``consecutive_probe_failures`` without re-reading state.
    """
    rd = config.review_dispatch
    quota = state.get("reviewer_quota") or {}
    consecutive_failures = int(quota.get("consecutive_probe_failures", 0)) + 1
    interval_minutes = rd.quota_probe_interval_minutes * (2 ** (consecutive_failures - 1))
    if rd.quota_probe_max_interval_minutes > 0:
        interval_minutes = min(interval_minutes, rd.quota_probe_max_interval_minutes)
    if reset_at is not None:
        # Back off until the provider's own stated reset, plus the resume
        # margin (provider resets are floors, not guarantees — issue #499).
        throttled_dt = reset_at + timedelta(seconds=config.runtime.throttle_resume_margin_s)
    else:
        throttled_dt = now_dt + timedelta(hours=rd.quota_reset_hours)
    throttled_until = throttled_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    probe_after = (
        (now_dt + timedelta(minutes=interval_minutes))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = set_reviewer_quota_exhausted(
        state, throttled_until=throttled_until, probe_after=probe_after
    )
    quota_record = {
        **state["reviewer_quota"],
        "consecutive_probe_failures": consecutive_failures,
        "reset_at": (
            reset_at.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            if reset_at is not None
            else None
        ),
    }
    state = {**state, "reviewer_quota": quota_record}
    return state, quota_record


def _merge_on_write_save(
    state_file: Path,
    state: dict[str, Any],
    write_gate: WriteGate,
    *,
    snapshot_prs: dict[str, Any],
    snapshot_reviewer_quota: Any,
    snapshot_events: list[dict[str, Any]],
    event_ring_cap: int,
) -> None:
    """Merge a sweep's computed changes onto fresh on-disk state and save
    (issue #594).

    Any ``dispatch_reviews`` sweep that loads a state snapshot via
    ``load_state_locked``, does slow work (filesystem or network I/O) with no
    lock held, then wants to persist what it computed must call this instead
    of a bare ``save_state(state_file, state)``. Writing the sweep's `state`
    wholesale would restore every field a concurrent writer (e.g. ``charlie
    unescalate``) changed in the gap between the snapshot load and this save
    to its pre-commit value, with no event explaining the reversal -- a
    classic read-modify-write lost update. ``unescalate`` itself already
    defends against this hazard in the other direction with the same
    re-load-and-re-apply-inside-the-lock pattern; this extends the same
    courtesy to the loop's sweeps.

    Re-loads fresh state under the lock and, per PR entry, applies only the
    fields that differ between ``state`` (the sweep's post-work view) and
    ``snapshot_prs`` (the entry-pass view) -- i.e. only the fields the sweep
    itself actually changed. Entries the sweep never touched are left exactly
    as fresh on-disk state has them, so a concurrent writer's commit to an
    untouched entry survives. For a PR the sweep DID touch, its overrides
    (e.g. a terminal ``status`` written because GitHub said the PR merged)
    still win over whatever a concurrent writer set on that same entry in the
    gap -- the sweep is the authority on the fact it just observed.

    The durable event audit is unaffected by any of this: every event was
    already dual-written to events.db via ``append_event(state_path=...)``
    before this call; this only keeps the bounded ``state.json`` event ring
    consistent by appending the sweep's new events onto the fresh ring.
    """
    write_gate = require_write_gate(write_gate)
    with state_lock(state_file):
        fresh = load_state(state_file)
        fresh_prs = dict(fresh.get("prs") or {})
        for pr_key, new_entry in (state.get("prs") or {}).items():
            if not isinstance(new_entry, dict):
                fresh_prs[pr_key] = new_entry
                continue
            if new_entry == snapshot_prs.get(pr_key):
                # Untouched by this sweep: preserve the fresh on-disk value so
                # a concurrent writer's commit survives.
                continue
            # Apply only the fields the sweep computed onto the fresh entry:
            # overrides = new_entry fields that differ from the entry-pass
            # snapshot. Fields the sweep did not touch stay as the fresh
            # on-disk value (preserving concurrent writes to them), and the
            # sweep's overrides win for the fields it legitimately owns.
            snapshot_entry = snapshot_prs.get(pr_key) or {}
            overrides = {
                field: value
                for field, value in new_entry.items()
                if value != snapshot_entry.get(field)
            }
            fresh_entry = fresh_prs.get(pr_key)
            fresh_prs[pr_key] = {
                **(fresh_entry if isinstance(fresh_entry, dict) else {}),
                **overrides,
            }
        merged: dict[str, Any] = dict(fresh)
        merged["prs"] = fresh_prs
        if state.get("reviewer_quota") != snapshot_reviewer_quota:
            merged["reviewer_quota"] = state["reviewer_quota"]
        # Identity diff, not a length-based slice: ``append_event`` always
        # rebuilds the events list via ``list(old) + [new]`` (never mutates
        # entries in place), so every snapshot event dict retains its
        # original ``id()`` for the life of this pass. When the ring is at
        # its cap (the normal steady-state -- 2000 entries by default), each
        # append truncates from the front, so the post-sweep list is the SAME
        # length as the snapshot and a length-based slice
        # ``current[len(snapshot):]`` wrongly evaluates to empty, silently
        # dropping this sweep's own events from the merged ring. Filtering by
        # identity against the snapshot's ids is correct regardless of
        # whether eviction happened.
        snapshot_event_ids = {id(e) for e in snapshot_events}
        sweep_appended_events = [
            e for e in (state.get("events") or []) if id(e) not in snapshot_event_ids
        ]
        if sweep_appended_events:
            ring = list(fresh.get("events") or []) + sweep_appended_events
            if len(ring) > event_ring_cap:
                ring = ring[-event_ring_cap:]
            merged["events"] = ring
        write_gate.save_state(merged)


class _ThrottleClassification(Enum):
    """Three-way classification of a dead reviewer's log for throttle
    detection (issue #1069).

    A plain bool cannot distinguish "the log was readable and contained no
    throttle marker" (``NOT_THROTTLED``) from "the log could not be read or
    was empty" (``UNDETERMINED``). Both used to collapse to
    ``throttled = False``, which burned the PR's dispatch attempt budget for
    a death that may not have been its fault AND failed to arm the fleet-wide
    reviewer-quota backoff — re-arming the #1342-1346 redispatch-into-the-wall
    outage mechanism through the read-error path rather than the logic path
    it was originally closed against.

    Folding the unknown into either existing branch is wrong: defaulting to
    ``THROTTLED`` over-applies fleet-wide backoff with no evidence (this file
    was separately burned by over-applying backoff), while defaulting to
    ``NOT_THROTTLED`` burns the per-PR attempt budget and leaves the fleet
    unprotected. The distinct third state carries its own handling — roll
    back the claim (preserving the attempt budget) without arming backoff.
    """

    THROTTLED = "throttled"
    NOT_THROTTLED = "not_throttled"
    UNDETERMINED = "undetermined"


def _detect_and_handle_stalled_reviews(
    reviews_dir: Path,
    state_file: Path,
    config: OrchestratorConfig,
    repo_root: Path,
    write_gate: WriteGate,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Detect reviewer processes that died without a verdict and free their claims.

    A reviewer is considered stalled/orphaned when its sidecar process is no
    longer alive and the claim timestamp is past the stale-claim timeout
    (``_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES``, currently 5 minutes -- see
    ``state.is_claim_stale``). When that happens, the per-PR
    ``review_dispatch_status`` is moved to ``review_dispatch_failed`` with the
    stale timestamp as ``review_dispatch_failed_at``. The next
    ``dispatch_reviews`` pass can then re-dispatch the PR after the same stale
    timeout elapses. Every reap path also tears down that PR's isolated review
    checkout (``worktree.remove_review_checkout``) so a dead reviewer never
    leaks its detached-HEAD checkout directory.

    A PR that reached ``status == "reviewing"`` but has no
    ``review_dispatch_status`` claim at all (a packet that was generated but
    never dispatched) is also reaped once its review packet is older than the
    stale-claim timeout. The claim is moved to ``review_dispatch_failed`` using
    the packet's own mtime as ``review_dispatch_failed_at`` so the next
    ``dispatch_reviews`` pass can retry it immediately.

    Callers should run ``_reap_review_verdicts`` first: it extracts and records
    any verdict a dead reviewer emitted, so by the time this sweep runs only
    PRs whose log has no parseable verdict fall through to the failed-claim
    retry/backoff path. This function is intentionally simpler than
    ``_detect_and_handle_stalled_sessions``: it performs claim/slot cleanup.

    ``now`` is the injectable clock (issue #828), forwarded to every
    ``is_claim_stale`` check in this sweep so every claim in one sweep is
    evaluated against a single consistent instant instead of each check
    independently racing the wall clock. It also seeds ``resolved_now``
    below, which both internal ``datetime.now(UTC)`` samples in the
    throttled-reviewer branch (the parsed-reset-time lookup and the
    quota-exhaustion backoff) resolve from, instead of each independently
    racing the wall clock. Defaults to ``datetime.now(UTC)`` when omitted, so
    production behavior is byte-identical; tests can freeze it and assert
    exact equality on the resulting ``throttled_until`` instead of a
    wall-clock-tolerance proximity check.
    """
    write_gate = require_write_gate(write_gate)
    resolved_now = now if now is not None else datetime.now(UTC)
    stalled: list[dict[str, Any]] = []
    sweep_events: list[tuple[str, dict[str, Any]]] = []
    state = load_state_locked(state_file)
    # Capture the entry-pass snapshot so the save block below can diff against
    # it and apply ONLY the entries/fields this sweep computed. Writing the
    # stale snapshot wholesale at save time would clobber any concurrent writer
    # (e.g. ``charlie unescalate``) that committed between this load and the
    # save -- a lost update across the sweep's read-modify-write window (issue
    # #594). The prs values are rebound (never mutated in place) by every
    # branch below, so a shallow copy of the mapping preserves the originals.
    snapshot_prs = dict(state.get("prs") or {})
    snapshot_reviewer_quota = state.get("reviewer_quota")
    snapshot_events = list(state.get("events") or [])
    changed = False
    seen_pr_keys: set[str] = set()
    # One provider-throttle condition per sweep, no matter how many dead
    # reviewers show the same limit signature: the exponential probe backoff
    # counts consecutive failed PROBES, and a wave of N simultaneously
    # throttled reviewers is one observation of the closed provider window,
    # not N. Without this, a 2-reviewer wave incremented the counter twice
    # per sweep and (combined with the un-reaped sidecars below) drove the
    # backoff from its 15-minute base to the 4-hour cap within 40 minutes
    # (observed live 2026-07-24: consecutive_probe_failures=14 from a single
    # quota outage).
    throttle_backoff_applied = False

    for w in iter_workers(reviews_dir):
        pr_key = str(w.issue_number)
        # A live reviewer needs no cleanup.
        if w.is_alive():
            continue
        # Respect the stale-claim timeout so a very recently dead reviewer is
        # not immediately re-dispatched (which can thrash if the underlying
        # launch path is flaky). Old dead reviewers become re-dispatchable.
        if not is_claim_stale(
            w.started_at, timeout_minutes=_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES, now=now
        ):
            continue

        seen_pr_keys.add(pr_key)
        pr_state = state["prs"].get(pr_key, {})
        status = pr_state.get("review_dispatch_status")
        if status in ("review_dispatch_completed", "review_dispatch_failed"):
            # Already terminal; don't overwrite a completed/failed record, but
            # reap the dead session's sidecar so it stops resurfacing here.
            w.reap_sidecar(reviews_dir)
            continue
        if status is None and pr_state.get("status") in ("merged", "closed"):
            # Lifecycle-reaped terminal PR: the claim was already cleared by
            # the orphan sweep. Without reaping the sidecar here, the dead
            # session resurrects as a phantom failed claim every pass and the
            # orphan sweep re-reaps it — an infinite stalled/reaped ping-pong
            # that floods the event ring (observed on 5 merged PRs, 07-22).
            w.reap_sidecar(reviews_dir)
            continue

        # A dead reviewer's own log may show why it died. If it hit a
        # provider throttle signature (e.g. "You've hit your session
        # limit"), the launch-time quota_hit check in dispatch_reviews never
        # saw it -- that check only fires when launch_claude_worker() itself
        # errors synchronously, but a throttled Claude Code CLI process
        # starts fine and only dies after printing the limit message to its
        # own log. Without this check, every pass here would mark the claim
        # review_dispatch_failed and the next dispatch_reviews pass would
        # relaunch straight into the same limit -- a redispatch loop that
        # runs every stale-claim interval for as long as the provider window
        # is closed, instead of backing off via the same reviewer-quota gate
        # the launch-time path uses (job-cannon PRs #1342/#1343/#1344/#1346,
        # 2026-07-21: 20+ hours of hot redispatch into a session-limit wall).
        #
        # Issue #1069: the log read itself can fail (OSError) or yield an
        # empty (0-byte) log -- a reviewer that died before its first flush.
        # Both used to collapse to ``throttled = False`` alongside a clean
        # read with no marker, indistinguishable from a genuine non-throttle
        # death. That burned the PR's attempt budget for a death that may not
        # have been its fault AND failed to arm the fleet-wide backoff,
        # re-arming the #1342-1346 outage mechanism through the read-error
        # path. The classification is now a three-way enum so the
        # undetermined case gets its own handling rather than being folded
        # into either existing branch (both defaults are wrong: defaulting to
        # throttled over-applies backoff with no evidence, defaulting to
        # not-throttled burns the budget and leaves the fleet unprotected).
        classification = _ThrottleClassification.NOT_THROTTLED
        log_mtime_dt: datetime | None = None
        reset_at: datetime | None = None
        log_read_ok = False
        try:
            log_file = Path(w.log_path)
            log_text = log_file.read_text(encoding="utf-8", errors="replace")
            log_read_ok = True
            try:
                log_mtime_dt = datetime.fromtimestamp(log_file.stat().st_mtime, tz=UTC)
            except OSError:
                log_mtime_dt = None
        except OSError:
            log_text = ""
            log_read_ok = False
        if not log_read_ok or not log_text:
            # Unreadable (OSError) or empty (0-byte) log: the reviewer died
            # before its first flush or the read raced an I/O hiccup. We
            # cannot determine whether the provider was throttled, so this is
            # the distinct third state -- not NOT_THROTTLED (which would burn
            # the PR's attempt budget for a death that may not be its fault)
            # and not THROTTLED (which would over-apply fleet-wide backoff
            # with no evidence). See _ThrottleClassification for the full
            # rationale.
            classification = _ThrottleClassification.UNDETERMINED
        else:
            tail = log_text[-2048:] if len(log_text) > 2048 else log_text
            matched = match_throttle_tail(tail, config.runtime.throttle_error_markers)[0]
            classification = (
                _ThrottleClassification.THROTTLED
                if matched
                else _ThrottleClassification.NOT_THROTTLED
            )
            # Issue #612: the session-limit notice names a specific reset
            # clock time in an IANA zone (e.g. "resets 1:20am
            # (America/Los_Angeles)"). Parse it once per dead session so the
            # fleet-wide backoff targets the provider's own stated reset
            # instead of a fixed quota_reset_hours guess. Only parsed on the
            # session that triggers the backoff (the first throttled one);
            # subsequent throttled sessions in the same wave reuse the
            # already-applied backoff, matching the one-increment-per-wave
            # guard below.
            if (
                classification is _ThrottleClassification.THROTTLED
                and not throttle_backoff_applied
            ):
                reset_at = parse_reset_clock_time(tail, resolved_now)

        if classification is _ThrottleClassification.THROTTLED:
            # A green flat-interval probe may have already cleared
            # reviewer_quota AFTER this reviewer died (issue #662): the
            # throttle signature in a dead session's log tail is frozen at
            # death time and does not reflect a recovery that happened
            # since. Re-applying backoff here would re-poison
            # reviewer_quota.throttled_until/probe_after anchored to "now"
            # rather than the original death time, delaying the next
            # dispatch by up to one probe cycle even though the quota
            # window is open. Suppress the backoff (but still roll back
            # the claim and reap the sidecar -- the reviewer is dead
            # regardless, and with the quota recovered the PR should be
            # immediately re-dispatchable) when a probe cleared after the
            # reviewer's last log write, which is the closest available
            # proxy for when the session died.
            probe_cleared_at = reviewer_quota_last_probe_cleared_at(state)
            backoff_suppressed = False
            if probe_cleared_at and log_mtime_dt is not None:
                try:
                    cleared_dt = datetime.fromisoformat(probe_cleared_at.replace("Z", "+00:00"))
                    if cleared_dt.tzinfo is None:
                        cleared_dt = cleared_dt.replace(tzinfo=UTC)
                    backoff_suppressed = cleared_dt > log_mtime_dt
                except (ValueError, TypeError):
                    backoff_suppressed = False
            if backoff_suppressed:
                throttled_until = None
            else:
                if not throttle_backoff_applied:
                    now_dt = resolved_now
                    state, quota_record = _set_reviewer_quota_exhausted_with_backoff(
                        state, config, now_dt, reset_at=reset_at
                    )
                    throttle_backoff_applied = True
                    # Distinct, queryable event for a quota-dead reviewer session
                    # (issue #612): carries the parsed reset time (or None when
                    # the notice carried no clock-time form / the zone was
                    # unavailable) so the condition is diagnosable as a quota
                    # exhaustion rather than collapsing into a generic
                    # "no verdict" / "provider_throttled" stall. Emitted once per
                    # sweep alongside the single backoff increment.
                    state = write_gate.append_event(
                        state,
                        "review_quota_exhausted",
                        {
                            "throttled_until": quota_record.get("throttled_until"),
                            "probe_after": quota_record.get("probe_after"),
                            "reset_at": quota_record.get("reset_at"),
                            "consecutive_probe_failures": quota_record.get(
                                "consecutive_probe_failures"
                            ),
                            "source": "stalled_review_sweep",
                        },
                    )
                throttled_until = state.get("reviewer_quota", {}).get("throttled_until")
            # A session that exhausted its full turn budget did real
            # PR-specific work -- its death is a PR-level outcome (the
            # review didn't fit the budget) regardless of what killed the
            # final API call. Only sessions that died WITHOUT such work
            # qualify for the provider-throttle rollback. _reap_review_verdicts
            # (which runs before this sweep) sets
            # review_turn_limit_summary_posted on the pr-state when it
            # extracted a turn-limit miss from this same dead session, and
            # that flag is reset to False on every new claim -- so it is the
            # single source of truth for "this dispatch lifecycle's session
            # did substantial work then died". When it is set, still apply
            # the global quota backoff (the throttle signal itself is real)
            # but treat the death as a normal counted failure so the per-PR
            # attempt cap can converge (issue #583: without this guard a PR
            # whose reviews deterministically hit the turn cap got unlimited
            # free retries whenever the account was near its limit, and the
            # 3-attempt cap never fired).
            if pr_state.get("review_turn_limit_summary_posted"):
                state["prs"][pr_key] = {
                    **pr_state,
                    "number": w.issue_number,
                    "review_dispatch_status": "review_dispatch_failed",
                    "review_dispatch_failed_at": w.started_at,
                    "review_dispatch_pending_at": None,
                    "review_dispatched_at": None,
                    "reviewer_pid": None,
                    "reviewer_process_start_time": None,
                    "review_log_unreadable_streak": 0,
                }
                event_payload = {
                    "pr_number": w.issue_number,
                    "pid": w.pid,
                    "started_at": w.started_at,
                    "reason": "provider_throttled_turn_limit_counted",
                    "throttled_until": throttled_until,
                    "backoff_suppressed": backoff_suppressed,
                }
                state = write_gate.append_event(
                    state,
                    "review_dispatch_stalled",
                    event_payload,
                    level=_classify_review_dispatch_stalled_level(event_payload),
                )
                changed = True
                stalled.append(
                    {
                        "pr": w.issue_number,
                        "pid": w.pid,
                        "started_at": w.started_at,
                        "reason": event_payload["reason"],
                    }
                )
                remove_review_checkout(repo_root, w.issue_number, reviews_dir=reviews_dir)
                w.reap_sidecar(reviews_dir)
                continue
            # Roll back (not fail) the claim: this is a global condition, not
            # a defect in this PR's review, so it should be immediately
            # re-dispatchable once the quota gate clears -- mirroring the
            # launch-time quota_hit rollback in dispatch_reviews. Also
            # decrement the attempt counter: the reviewer hit a provider
            # limit, not a PR-specific failure, so this must not consume the
            # per-PR dispatch attempt budget.
            rolled_back = without_review_dispatch_claim(pr_state)
            attempt_count = int(pr_state.get("review_dispatch_attempt_count", 0))
            if attempt_count > 0:
                rolled_back["review_dispatch_attempt_count"] = attempt_count - 1
            rolled_back["review_log_unreadable_streak"] = 0
            state["prs"][pr_key] = rolled_back
            event_payload = {
                "pr_number": w.issue_number,
                "pid": w.pid,
                "started_at": w.started_at,
                "reason": "provider_throttled",
                "throttled_until": throttled_until,
                "backoff_suppressed": backoff_suppressed,
            }
            state = write_gate.append_event(
                state,
                "review_dispatch_stalled",
                event_payload,
                level=_classify_review_dispatch_stalled_level(event_payload),
            )
            changed = True
            stalled.append(
                {
                    "pr": w.issue_number,
                    "pid": w.pid,
                    "started_at": w.started_at,
                    "reason": event_payload["reason"],
                }
            )
            remove_review_checkout(repo_root, w.issue_number, reviews_dir=reviews_dir)
            # Reap the sidecar like every other handled path in this sweep.
            # The rolled-back claim is deliberately non-terminal (so the PR
            # re-dispatches once the quota gate clears), which means neither
            # terminal guard above will ever reap this sidecar -- without
            # this line the same dead reviewer resurfaces every sweep, its
            # log tail still matches the throttle signature, and each pass
            # re-applies the exponential backoff for a session that died
            # exactly once (observed live 2026-07-24: two dead reviewers
            # re-counted across ~6 passes pushed probe_after 4 hours out
            # while the provider window was already open again).
            w.reap_sidecar(reviews_dir)
            continue

        if classification is _ThrottleClassification.UNDETERMINED:
            # The reviewer's log could not be read (OSError) or was empty
            # (0-byte, died before first flush). We cannot tell whether the
            # provider was throttled, so this is neither a counted PR-level
            # failure (which would burn the attempt budget for a death that
            # may not be the PR's fault) nor a confirmed throttle (which
            # would arm fleet-wide backoff with no evidence -- this file was
            # burned by over-applying backoff before).
            #
            # Review finding on PR #1161: the original fix rolled back the
            # claim and decremented the attempt counter on every UNDETERMINED
            # death, exactly like the throttle path. But unlike the throttle
            # path it did NOT arm fleet-wide backoff, so nothing stopped the
            # redispatch loop: dispatch increments the counter, UNDETERMINED
            # decrements it, net zero per cycle, and the
            # ``max_review_dispatch_attempts`` cap never fired -- an
            # unbounded, unthrottled redispatch loop for any PR whose
            # reviewer log stays persistently unreadable/empty, the same
            # outage shape as #1342-1346 via a new path.
            #
            # The bound: track a per-PR ``review_log_unreadable_streak``
            # across sweeps. The first N consecutive UNDETERMINED deaths
            # (``max_consecutive_review_log_unreadable``, default 3) are
            # treated as transient I/O hiccups -- roll back the claim and
            # decrement the attempt counter, exactly like the throttle path
            # but without arming backoff. Once the streak exceeds N the
            # condition is persistent, not transient: subsequent UNDETERMINED
            # deaths become counted failures (attempt counter NOT
            # decremented, status set to ``review_dispatch_failed``) so the
            # existing ``max_review_dispatch_attempts`` cap converges and
            # escalates instead of looping forever. The streak resets on any
            # definitive outcome (throttled, not-throttled, verdict recorded,
            # new packet, operator unescalate).
            max_unreadable_streak = config.review_dispatch.max_consecutive_review_log_unreadable
            prev_streak = int(pr_state.get("review_log_unreadable_streak", 0))
            streak = prev_streak + 1
            if max_unreadable_streak > 0 and streak > max_unreadable_streak:
                # Persistent unreadable-log condition: stop preserving the
                # attempt budget. This is a counted failure (like the
                # NOT_THROTTLED path below) -- the attempt counter is NOT
                # decremented so the existing ``max_review_dispatch_attempts``
                # cap can fire and escalate. A distinct event reason keeps
                # the persistent condition diagnosable separately from a
                # one-off unreadable death (issue #1069).
                state["prs"][pr_key] = {
                    **pr_state,
                    "number": w.issue_number,
                    "review_dispatch_status": "review_dispatch_failed",
                    "review_dispatch_failed_at": w.started_at,
                    "review_dispatch_pending_at": None,
                    "review_dispatched_at": None,
                    "reviewer_pid": None,
                    "reviewer_process_start_time": None,
                    "review_log_unreadable_streak": streak,
                }
                event_payload = {
                    "pr_number": w.issue_number,
                    "pid": w.pid,
                    "started_at": w.started_at,
                    "reason": "review_log_persistently_unreadable",
                    "streak": streak,
                }
                state = write_gate.append_event(
                    state,
                    "review_dispatch_stalled",
                    event_payload,
                    level=_classify_review_dispatch_stalled_level(event_payload),
                )
                changed = True
                stalled.append(
                    {
                        "pr": w.issue_number,
                        "pid": w.pid,
                        "started_at": w.started_at,
                        "reason": event_payload["reason"],
                    }
                )
                remove_review_checkout(repo_root, w.issue_number, reviews_dir=reviews_dir)
                w.reap_sidecar(reviews_dir)
                continue
            # Transient unreadable-log death (streak <= N): roll back the
            # claim and decrement the attempt counter exactly like the
            # throttle path, but do NOT arm the reviewer-quota backoff. If
            # the provider IS throttled, the next dispatch launches a new
            # reviewer whose readable log will classify correctly and arm
            # backoff on the next sweep; if it is not, the PR re-dispatches
            # without burning its budget (issue #1069).
            rolled_back = without_review_dispatch_claim(pr_state)
            attempt_count = int(pr_state.get("review_dispatch_attempt_count", 0))
            if attempt_count > 0:
                rolled_back["review_dispatch_attempt_count"] = attempt_count - 1
            rolled_back["review_log_unreadable_streak"] = streak
            state["prs"][pr_key] = rolled_back
            event_payload = {
                "pr_number": w.issue_number,
                "pid": w.pid,
                "started_at": w.started_at,
                "reason": "review_log_unreadable",
                "streak": streak,
            }
            state = write_gate.append_event(
                state,
                "review_dispatch_stalled",
                event_payload,
                level=_classify_review_dispatch_stalled_level(event_payload),
            )
            changed = True
            stalled.append(
                {
                    "pr": w.issue_number,
                    "pid": w.pid,
                    "started_at": w.started_at,
                    "reason": event_payload["reason"],
                }
            )
            remove_review_checkout(repo_root, w.issue_number, reviews_dir=reviews_dir)
            # Reap the sidecar: the rolled-back claim is non-terminal, so
            # neither terminal guard above will ever reap it -- without this
            # the same dead reviewer resurfaces every sweep (same rationale
            # as the throttle path's reap above).
            w.reap_sidecar(reviews_dir)
            continue

        state["prs"][pr_key] = {
            **pr_state,
            "number": w.issue_number,
            "review_dispatch_status": "review_dispatch_failed",
            "review_dispatch_failed_at": w.started_at,
            "review_dispatch_pending_at": None,
            "review_dispatched_at": None,
            "reviewer_pid": None,
            "reviewer_process_start_time": None,
            "review_log_unreadable_streak": 0,
        }
        sweep_events.append(
            (
                "review_dispatch_stalled",
                {
                    "pr_number": w.issue_number,
                    "pid": w.pid,
                    "started_at": w.started_at,
                },
            )
        )
        changed = True
        stalled.append({"pr": w.issue_number, "pid": w.pid, "started_at": w.started_at})
        state, _ = _remove_review_checkout_with_warning(
            state,
            repo_root,
            reviews_dir,
            w.issue_number,
            write_gate=write_gate,
            state_file=state_file,
        )
        # The failed record above is now the source of truth for redispatch;
        # the dead session's sidecar must go with the checkout or it re-enters
        # this sweep as a phantom on every subsequent pass.
        w.reap_sidecar(reviews_dir)

    # Catch state entries that have no sidecar (launch crashed before sidecar
    # write, or sidecar was deleted) and are past the stale timeout.
    for pr_key, pr_state in list(state.get("prs", {}).items()):
        if pr_key in seen_pr_keys or not isinstance(pr_state, dict):
            continue

        status = pr_state.get("review_dispatch_status")
        if status == "review_dispatch_pending":
            pending_at = pr_state.get("review_dispatch_pending_at")
            if pending_at and is_claim_stale(
                pending_at, timeout_minutes=_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES, now=now
            ):
                state["prs"][pr_key] = {
                    **pr_state,
                    "review_dispatch_status": "review_dispatch_failed",
                    "review_dispatch_failed_at": pending_at,
                    "review_dispatch_pending_at": None,
                    "reviewer_pid": None,
                    "reviewer_process_start_time": None,
                }
                sweep_events.append(
                    (
                        "review_dispatch_stalled",
                        {
                            "pr_number": int(pr_key) if pr_key.isdigit() else None,
                            "status": "pending",
                            "pending_at": pending_at,
                        },
                    )
                )
                changed = True
                stalled.append(
                    {
                        "pr": int(pr_key) if pr_key.isdigit() else None,
                        "pending_at": pending_at,
                    }
                )
                if pr_key.isdigit():
                    state, _ = _remove_review_checkout_with_warning(
                        state,
                        repo_root,
                        reviews_dir,
                        int(pr_key),
                        write_gate=write_gate,
                        state_file=state_file,
                    )
        elif status == "review_dispatch_dispatched":
            reviewer_pid = pr_state.get("reviewer_pid")
            process_start_time = pr_state.get("reviewer_process_start_time")
            pid_alive = reviewer_pid is not None and is_pid_alive(reviewer_pid, process_start_time)
            if pid_alive:
                continue
            dispatched_at = pr_state.get("review_dispatched_at")
            if dispatched_at and is_claim_stale(
                dispatched_at, timeout_minutes=_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES, now=now
            ):
                state["prs"][pr_key] = {
                    **pr_state,
                    "review_dispatch_status": "review_dispatch_failed",
                    "review_dispatch_failed_at": dispatched_at,
                    "review_dispatched_at": None,
                    "reviewer_pid": None,
                    "reviewer_process_start_time": None,
                }
                sweep_events.append(
                    (
                        "review_dispatch_stalled",
                        {
                            "pr_number": int(pr_key) if pr_key.isdigit() else None,
                            "status": "dispatched",
                            "dispatched_at": dispatched_at,
                        },
                    )
                )
                changed = True
                stalled.append(
                    {
                        "pr": int(pr_key) if pr_key.isdigit() else None,
                        "dispatched_at": dispatched_at,
                    }
                )
                if pr_key.isdigit():
                    state, _ = _remove_review_checkout_with_warning(
                        state,
                        repo_root,
                        reviews_dir,
                        int(pr_key),
                        write_gate=write_gate,
                        state_file=state_file,
                    )
        elif status is None and pr_state.get("status") == "reviewing":
            # Issue #487: a review packet was generated but was never claimed or
            # dispatched at all. If the packet is past the stale-claim timeout,
            # move the (missing) claim to failed using the packet's own mtime as
            # the failure timestamp so the next dispatch_reviews pass can retry.
            prompt_path_str = pr_state.get("prompt_path")
            if not prompt_path_str:
                # Issue #708: record that this PR's stale-claim recovery was
                # skipped so a future stuck-PR investigation does not require
                # re-deriving from source whether recovery ran or silently gave
                # up. log_event writes directly to events.db (not the state.json
                # events ring) because this skip does not mutate state -- routing
                # it through sweep_events/_append_sweep_events would only persist
                # when ``changed`` is set by an unrelated sibling PR in the same
                # pass, reintroducing the silent-drop under exactly the low-load
                # conditions where the signal matters most.
                log_event(
                    state_file,
                    "review_stale_claim_recovery_skipped",
                    {
                        "pr_number": int(pr_key) if pr_key.isdigit() else None,
                        "reason": "prompt_path missing from state",
                    },
                    level="warning",
                )
                continue
            prompt_path = Path(prompt_path_str)
            if not prompt_path.exists():
                log_event(
                    state_file,
                    "review_stale_claim_recovery_skipped",
                    {
                        "pr_number": int(pr_key) if pr_key.isdigit() else None,
                        "reason": "prompt_path file does not exist on disk",
                        "prompt_path": prompt_path_str,
                    },
                    level="warning",
                )
                continue

            decision_value: str | None = "missing"
            decision_path_str = pr_state.get("decision_path")
            if decision_path_str:
                decision_path = Path(decision_path_str)
                if decision_path.exists():
                    try:
                        with decision_path.open("r", encoding="utf-8") as handle:
                            decision_data = json.load(handle)
                        if isinstance(decision_data, dict):
                            decision_value = decision_data.get("decision")
                    except (OSError, json.JSONDecodeError):
                        decision_value = "invalid"
            if decision_value not in ("pending", "missing", "invalid", None):
                # Issue #734: the decision_path gate is the second of three
                # silent skip paths in stale-claim recovery. A PR whose review
                # already produced a verdict (e.g. ``request_changes``) but is
                # still in ``reviewing`` status is passed over every pass with
                # no trace -- the verdict was never acted upon, and without this
                # event nobody can tell recovery considered the PR and declined.
                # Same ``log_event`` / ``review_stale_claim_recovery_skipped``
                # pattern as the prompt_path skips above (issue #708): writes
                # directly to events.db because this skip does not mutate state.
                log_event(
                    state_file,
                    "review_stale_claim_recovery_skipped",
                    {
                        "pr_number": int(pr_key) if pr_key.isdigit() else None,
                        "reason": "decision_already_recorded",
                        "decision": decision_value,
                    },
                    level="warning",
                )
                continue

            prompt_mtime = prompt_path.stat().st_mtime
            packet_age = (
                datetime.fromtimestamp(prompt_mtime, tz=UTC)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
            if not is_claim_stale(
                packet_age, timeout_minutes=_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES, now=now
            ):
                # Issue #734: the packet-age gate is the third silent skip path.
                # The packet is not stale yet, so recovery defers to a future
                # pass. This is normal flow control, not a failure -- but the
                # issue requires all three exits to be observable so a stuck-PR
                # investigation can confirm recovery ran and chose to defer
                # rather than silently doing nothing. ``level="info"`` because
                # this is expected behavior (the packet simply is not old
                # enough), unlike the other two skips which indicate a PR that
                # recovery cannot help.
                log_event(
                    state_file,
                    "review_stale_claim_recovery_skipped",
                    {
                        "pr_number": int(pr_key) if pr_key.isdigit() else None,
                        "reason": "packet_not_stale",
                        "packet_age": packet_age,
                    },
                    level="info",
                )
                continue

            state["prs"][pr_key] = {
                **pr_state,
                "number": int(pr_key) if pr_key.isdigit() else pr_state.get("number"),
                "review_dispatch_status": "review_dispatch_failed",
                "review_dispatch_failed_at": packet_age,
                "review_dispatch_pending_at": None,
                "review_dispatched_at": None,
                "reviewer_pid": None,
                "reviewer_process_start_time": None,
            }
            # Issue #748: an unclaimed packet is a transient startup race, not
            # a terminal failure. ``review()`` generated the packet but
            # ``dispatch_reviews`` has not claimed it yet; this sweep marks it
            # ``review_dispatch_failed`` so the next dispatch pass retries.
            # Measured against events.db (81 unclaimed events, 2026-07-23 to
            # 2026-08-13): 31/81 (38%) recovered via ``review_dispatch_claim``
            # with a median of 19s (27/31 under 60s); the remaining 50/81 were
            # handled by fallback mechanisms (orphaned-worker routing, stale-
            # claim reaping, escalation) rather than the normal dispatch path.
            # In neither case is the packet stuck -- the sweep's
            # ``review_dispatch_failed`` transition guarantees a retry path.
            # Routing this through ``append_event`` directly with
            # ``level="warning"`` (instead of ``sweep_events``) mirrors the two
            # ``provider_throttled*`` paths above and keeps it out of the
            # ``_append_sweep_events`` batcher, which deliberately does not
            # classify levels (see its comment).
            unclaimed_payload = {
                "pr_number": int(pr_key) if pr_key.isdigit() else None,
                "status": "unclaimed",
                "prompt_mtime": packet_age,
            }
            state = write_gate.append_event(
                state,
                "review_dispatch_stalled",
                unclaimed_payload,
                level="warning",
            )
            changed = True
            stalled.append(
                {
                    "pr": int(pr_key) if pr_key.isdigit() else None,
                    "prompt_mtime": packet_age,
                }
            )
            if pr_key.isdigit():
                state, _ = _remove_review_checkout_with_warning(
                    state,
                    repo_root,
                    reviews_dir,
                    int(pr_key),
                    write_gate=write_gate,
                    state_file=state_file,
                )

    if changed:
        state = _append_sweep_events(
            state,
            sweep_events,
            max_size=config.runtime.event_ring_size,
            state_file=state_file,
            write_gate=write_gate,
        )
        # Merge-on-write (issue #594) -- see ``_merge_on_write_save`` for why a
        # bare ``save_state(state_file, state)`` here would clobber a
        # concurrent writer (e.g. ``charlie unescalate``).
        _merge_on_write_save(
            state_file,
            state,
            write_gate=write_gate,
            snapshot_prs=snapshot_prs,
            snapshot_reviewer_quota=snapshot_reviewer_quota,
            snapshot_events=snapshot_events,
            event_ring_cap=config.runtime.event_ring_size,
        )

    return stalled


def _reap_review_sidecar(reviews_dir: Path, pr_number: int) -> None:
    """Delete a dead reviewer's sidecar so it cannot resurrect as a phantom.

    Reviewer sessions are keyed by PR number in ``reviews_dir``. Never touches
    a live session's sidecar (the governor counts live sidecars — deleting one
    would silently free a slot for over-cap dispatch). Best-effort:
    ``WorkerView.reap_sidecar`` swallows OSError, and a sidecar that survives
    one pass is reaped on the next.
    """
    for w in iter_workers(reviews_dir):
        if w.issue_number == pr_number and not w.is_alive():
            w.reap_sidecar(reviews_dir)


def _reap_completed_review_checkouts(
    repo_root: Path,
    reviews_dir: Path,
    state_file: Path,
) -> list[int]:
    """Remove isolated review checkouts for PRs whose reviewer already
    recorded a verdict, once the reviewer process itself has exited.

    ``record_review`` (called in-process by ``_reap_review_verdicts`` or invoked
    via the ``verdict`` CLI) sets ``review_dispatch_status = "review_dispatch_completed"`` and
    clears ``reviewer_pid``/``reviewer_process_start_time`` on state.json as
    part of recording the verdict — so by the time this sweep can see
    "completed", state.json itself no longer carries a PID to check liveness
    against. The claude-code sidecar in ``reviews_dir`` still does (it isn't
    touched by ``record_review``), so liveness is checked via ``iter_workers``
    instead. This avoids removing the checkout while the reviewer session
    that just wrote the verdict is still in the process of exiting.
    """
    state = load_state_locked(state_file)
    completed_prs = {
        int(pr_key)
        for pr_key, entry in state.get("prs", {}).items()
        if isinstance(entry, dict)
        and entry.get("review_dispatch_status") == "review_dispatch_completed"
        and pr_key.isdigit()
    }
    if not completed_prs:
        return []

    alive_pr_numbers = _alive_review_worker_issue_numbers(reviews_dir)
    reaped: list[int] = []
    for pr_number in sorted(completed_prs):
        if pr_number in alive_pr_numbers:
            continue
        if remove_review_checkout(repo_root, pr_number, reviews_dir=reviews_dir):
            reaped.append(pr_number)
        # The verdict is recorded and the reviewer has exited: the sidecar has
        # served its purpose and must not linger as a phantom session.
        _reap_review_sidecar(reviews_dir, pr_number)
    return reaped


def _reap_orphaned_review_checkouts(
    gh: GitHubLike,
    repo_root: Path,
    reviews_dir: Path,
    state_file: Path,
    config: OrchestratorConfig,
    write_gate: WriteGate,
) -> list[int]:
    """Remove isolated review checkouts for PRs whose GitHub lifecycle has
    already reached ``MERGED`` or ``CLOSED`` and whose reviewer process has
    exited.

    This is the review-dispatch-pass counterpart to reconcile.py's
    ``merged_outside_orchestrator`` and ``closed_unmerged_pr_active_labels``
    drift handlers. It runs unconditionally at the top of ``dispatch_reviews``
    so an externally-merged/closed PR never leaves its ``reviews_dir``
    checkout or ``review_dispatch_*`` claim alive indefinitely. A PR whose
    reviewer sidecar is still alive is deferred to a later pass so the live
    session is not interrupted.
    """
    write_gate = require_write_gate(write_gate)
    state = load_state_locked(state_file)
    # Capture the entry-pass snapshot so the save block below can merge-on-write
    # instead of restoring this stale snapshot wholesale (issue #594). The
    # candidate loop below does a ``gh.pr_view`` network call per candidate PR,
    # potentially many, before the save -- the same shape of unlocked
    # read-modify-write window that let a concurrent ``charlie unescalate``
    # get silently reverted in ``_detect_and_handle_stalled_reviews``. ``prs``
    # entries are rebound (never mutated in place) below, so a shallow copy of
    # the mapping preserves the originals.
    snapshot_prs = dict(state.get("prs") or {})
    snapshot_reviewer_quota = state.get("reviewer_quota")
    snapshot_events = list(state.get("events") or [])
    candidate_pr_numbers: set[int] = set()

    review_dispatch_keys = (
        "review_dispatch_status",
        "review_dispatch_pending_at",
        "review_dispatched_at",
        "review_dispatch_failed_at",
        "reviewer_pid",
        "reviewer_process_start_time",
    )
    for pr_key, entry in state.get("prs", {}).items():
        if not isinstance(entry, dict):
            continue
        if any(entry.get(field) is not None for field in review_dispatch_keys):
            try:
                candidate_pr_numbers.add(int(pr_key))
            except ValueError:
                continue

    if reviews_dir.is_dir():
        for entry in reviews_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("pr-"):
                suffix = entry.name.split("-", 1)[1]
                try:
                    candidate_pr_numbers.add(int(suffix))
                except ValueError:
                    continue

    if not candidate_pr_numbers:
        return []

    alive_pr_numbers = _alive_review_worker_issue_numbers(reviews_dir)
    reaped: list[int] = []
    sweep_events: list[tuple[str, dict[str, Any]]] = []
    changed = False
    for pr_number in sorted(candidate_pr_numbers):
        try:
            pr = gh.pr_view(pr_number)
        except Exception:
            # PR not found or transient lookup failure; do not act on uncertainty.
            continue
        if not isinstance(pr, dict):
            continue

        gh_state = str(pr.get("state") or "").upper()
        if gh_state not in ("MERGED", "CLOSED"):
            continue

        # Issue #504: a live reviewer process keeps its checkout alive until it exits.
        if pr_number in alive_pr_numbers:
            continue

        pr_key = str(pr_number)
        pr_state = state["prs"].get(pr_key, {})
        new_pr_state = without_review_dispatch_claim(pr_state)
        new_pr_state["number"] = pr_number
        if gh_state == "MERGED":
            new_pr_state["status"] = "merged"
            # Issue #747: stamp ``merged_at`` only on a genuine non-merged ->
            # merged transition so re-reaping a PR already recorded as merged
            # does not overwrite the original observation time.
            if pr_state.get("status") != "merged":
                new_pr_state["merged_at"] = utc_now()
        else:
            # Record the terminal closed state so a future pass does not
            # re-query.  Always overwrite — a stale "reviewing" status left
            # by the review pipeline causes the unclaimed-stalled sweep to
            # re-trigger every pass (infinite ping-pong with this reaper).
            new_pr_state["status"] = "closed"
        state["prs"][pr_key] = new_pr_state
        changed = True

        state, removed = _remove_review_checkout_with_warning(
            state,
            repo_root,
            reviews_dir,
            pr_number,
            write_gate=write_gate,
            state_file=state_file,
        )
        # Reap the sidecar with the checkout: leaving it resurrects the dead
        # session as a phantom failed claim in the stalled sweep next pass,
        # which re-triggers this reap — an infinite ping-pong per merged PR.
        _reap_review_sidecar(reviews_dir, pr_number)
        if removed:
            sweep_events.append(
                (
                    "review_dispatch_lifecycle_reaped",
                    {"pr_number": pr_number, "github_state": gh_state.lower()},
                )
            )
            reaped.append(pr_number)

    if changed:
        state = _append_sweep_events(
            state,
            sweep_events,
            max_size=config.runtime.event_ring_size,
            state_file=state_file,
            write_gate=write_gate,
        )
        # Merge-on-write (issue #594) -- see ``_merge_on_write_save`` for why a
        # bare ``save_state(state_file, state)`` here would clobber a
        # concurrent writer (e.g. ``charlie unescalate``). A PR this reap DID
        # touch keeps its terminal ``status`` ("merged"/"closed") regardless
        # of what a concurrent writer set on that same entry in the gap --
        # GitHub's lifecycle state is authoritative once observed.
        _merge_on_write_save(
            state_file,
            state,
            write_gate=write_gate,
            snapshot_prs=snapshot_prs,
            snapshot_reviewer_quota=snapshot_reviewer_quota,
            snapshot_events=snapshot_events,
            event_ring_cap=config.runtime.event_ring_size,
        )

    return reaped


def _classify_review_dispatch_stalled_level(payload: dict[str, Any]) -> str | None:
    """Return ``warning`` if a ``review_dispatch_stalled`` payload is a throttle reason.

    Genuine stalls (no ``reason`` key, or a non-throttle ``status``/``reason``
    value) return ``None`` so the event falls back to the registry's ``error``
    default. This centralizes the "classify by reason" decision in one place.
    """
    reason = payload.get("reason")
    if isinstance(reason, str) and reason.startswith("provider_throttled"):
        return "warning"
    return None


def _append_sweep_events(
    state: dict[str, Any],
    sweep_events: list[tuple[str, dict[str, Any]]],
    max_size: int | None = None,
    *,
    state_file: Path | None = None,
    write_gate: WriteGate,
) -> dict[str, Any]:
    """Append events collected during a sweep, aggregating same-kind runs.

    A single occurrence of a kind is emitted with the original kind and payload.
    Multiple occurrences of the same kind are emitted as one ``{kind}_sweep`` event
    with a count and a numbers list. This prevents a single bulk sweep from
    flooding the bounded event buffer and evicting unrelated diagnostic history.

    Issue #1264 (W6 PR3, R5 completion): both ``append_event`` calls below now
    go through ``write_gate``, which auto-binds ``state_path`` from its own
    ``state_path`` field. ``state_file`` is consequently now vestigial at
    every call site (equal to ``write_gate.state_path`` in every current
    caller, but unenforced) -- kept as a parameter rather than removed, the
    same call PR2 made for the analogous ``_merge_on_write_save`` seam
    (adversarial-review finding F3), to avoid churning callers for no
    behavioral gain.
    """
    write_gate = require_write_gate(write_gate)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for kind, payload in sweep_events:
        grouped.setdefault(kind, []).append(payload)

    # Deliberately no level= classification here. Every payload routed through
    # this batching path carries `status` (pending/dispatched/unclaimed) or no
    # reason at all -- the two `provider_throttled*` reasons call append_event
    # directly and are classified at those call sites. Adding a level= here
    # would also suppress test_event_kind_registry_exhaustive's coverage of this
    # function's unresolvable `kind` loop variable, since a site that declares
    # its own level is exempt from kind verification. See #1029.
    for kind, payloads in grouped.items():
        if len(payloads) == 1:
            # kind is the loop variable over sweep_events, a list built from
            # `sweep_events.append((literal_kind, payload))` call sites elsewhere in this
            # file -- those literals are the ones actually checked for consumers.
            state = write_gate.append_event(
                state, kind, payloads[0], max_size=max_size
            )  # event-consumer: pointer -- literal chosen at each sweep_events.append(...) site
        else:
            numbers: list[int] = []
            numbers_key = "numbers"
            for payload in payloads:
                if payload.get("issue_number") is not None:
                    numbers.append(payload["issue_number"])
                    numbers_key = "issue_numbers"
                elif payload.get("pr_number") is not None:
                    numbers.append(payload["pr_number"])
                    if numbers_key == "numbers":
                        numbers_key = "pr_numbers"
                elif payload.get("number") is not None:
                    numbers.append(payload["number"])
            state = write_gate.append_event(
                state,
                # event-consumer: pointer -- same kind loop variable as the single-payload
                # branch above, with the `_sweep` suffix; literals are scanned at their
                # sweep_events.append(...) build sites elsewhere in this file
                f"{kind}_sweep",
                {
                    "count": len(payloads),
                    numbers_key: numbers,
                },
                max_size=max_size,
            )
    return state
