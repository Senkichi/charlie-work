"""Dead-worker/session-reap free-function family (issue #1317).

Extracted verbatim from ``workflow.py``: the dead-worker/session half of the
35-member reap family originally scoped by issue #1283's A6 recon. The
call-graph-clean stalled-review half (``stalled_review_reap.py``, 10 units)
shipped separately as A6 proper; this module carries the dead-worker/session
half, spun off to its own issue per the operator's staged-split decision
(#1283, 2026-08-17 comment) because that half is entangled with
``_detect_and_handle_orphaned_workers`` -- at 1,384 physical lines (grown
from 1,203L at #1283's A6 recon time) a single function far over this
repo's 800-line module cap, whose decomposition #1283/#1317 both treat as a
deliberate refactor, not a verbatim move.

**`_detect_and_handle_orphaned_workers` deliberately stays in `workflow.py`.**
Per #1317's own text ("Requirements for the refactor: plan the
`_detect_and_handle_orphaned_workers` decomposition as a deliberate design
exercise... before any code moves") and #1283's recorded staged-split
decision (which explicitly rejected shipping the dead-worker half as a
blanket-cap-exemption verbatim move, the option this repo took for the
call-graph-clean stalled-review half instead), this extraction moves the
*rest* of the family -- the 23 clean call-graph units below it depends on,
plus one pure-utility peer (`_is_pr_updated_at_older_than`) required to
avoid a `workflow.py` <-> `dead_worker_reap.py` import cycle -- and leaves
the 1,384-line function itself behind. It keeps calling every moved name
unchanged: those calls resolve through `workflow.py`'s own globals, which
this module's facade re-export block repopulates, exactly like every other
Phase-A extraction's callers elsewhere in `workflow.py`.

Members (25 functions plus 2 threshold constants, in original file order):
``_is_startup_death``,
``_worker_death_bounded_runtime_seconds`` (plus the
``STARTUP_DEATH_THRESHOLD_SECONDS`` constant both share),
``_session_failed_relabeled_payload``, ``_emit_session_failed_relabeled``,
``_count_live_sessions``, ``_detect_stalled_sessions``,
``_detect_and_handle_stalled_sessions``, ``_worker_pid_alive``,
``_orphan_head_fingerprint``, ``_is_zero_artifact_dispatch_loop`` (plus its
``_ZERO_ARTIFACT_ESCALATION_THRESHOLD`` constant),
``_sweep_orphan_processes_for_dead_sessions``, ``_log_worker_census``,
``_rework_pr_for_worker``, ``_reap_restore_rework_requested``,
``_is_pr_updated_at_older_than``, ``_is_pre_review_rework_candidate``,
``_route_dead_worker_to_pre_review_rework``,
``_classify_dead_sessions_and_update_throttle_state``, ``_safe_repo_slug``,
``_dispatching_repo_name``, ``_open_salvage_pr``, ``_salvage_already_landed``,
``_attempt_salvage``, ``_open_pr_for_orphaned_branch``,
``_issues_with_live_workers``.

Judgment calls on membership (call-graph derived, not name-guessed --
disclosed per this issue's own instructions):

- ``_is_pr_updated_at_older_than`` is call-graph-shared: two of its three
  real callers (``_is_readiness_no_ci_stall``, a direct merge-lane call)
  are auto-merge/readiness logic that stays in ``workflow.py``, not this
  family. It moves anyway because its third caller,
  ``_is_pre_review_rework_candidate``, is a genuine family member and
  leaving the utility behind would force ``_is_pre_review_rework_candidate``
  to import back from ``workflow.py`` -- an import cycle. Its non-family
  callers keep working via the facade re-export below, the same as every
  other multi-caller symbol every prior Phase-A extraction has moved.
- ``_dispatching_repo_name`` is call-graph-shared the same way: used by two
  dispatch-selection-adjacent ``OrchestratorApp`` methods that stay in
  ``workflow.py``, and by two family members
  (``_classify_dead_sessions_and_update_throttle_state``,
  ``_detect_and_handle_orphaned_workers`` itself). Same cycle-avoidance
  rationale as above.
- ``_is_readiness_no_ci_stall``, ``_has_other_open_pr``,
  ``_is_rerun_already_running_error``, ``_format_merge_attempt_alarm_message``,
  ``_format_stale_base_alarm_message``, ``_is_pending_only``,
  ``_build_attention_digest``, and the shared utilities ``slugify`` /
  ``_parse_iso_timestamp`` are name- or position-adjacent but are NOT
  members: their real callers are the auto-merge/readiness lane or other
  disconnected domains, confirmed by grep across every call site, not by
  the naming pattern that misled earlier Phase-A recon passes (per #1283's
  own recorded corrections for A1-A5).

This module is expected to exceed the repo's normal 800-line-per-module
cap; consistent with the `stalled_review_reap.py` precedent (#1283,
operator decision 2026-08-17), that gate is waived here given the
byte-identical-move discipline this preserves.

``workflow.py`` re-exports every symbol here via a facade import block
(mirroring `config.py`'s `RunnerAllocationConfig` re-export pattern and
this repo's own `dispatch_selection.py` / `escalation.py` /
`verdict_parsing.py` / `rework_prompts.py` / `ci_findings.py` /
`backlog_reachability.py` / `stalled_review_reap.py` precedents), so
existing import paths and monkeypatch targets keep working unchanged.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .closing_reference import (
    ValidationResult,
    closing_issues_referenced_numbers,
    validate_closing_reference,
)
from .config import (
    DETERMINISTIC_ESCALATION_FAILURE_KINDS,
    DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS,
    OrchestratorConfig,
)
from .cross_repo_gate import cross_repo_scope_gate
from .dispatch_selection import _windowed_redispatch_at, _windowed_worker_death_at
from .escalation import _escalate_issue, _escalation_edge
from .fleet_registry import managed_repo_names
from .github import (
    GitHubLike,
    PR_CLOSING_ISSUES_FIELDS,
    build_branch_issue_validator,
    label_names,
    linked_issue_number,
)
from .instrumentation import log_event
from .labels import TransitionOutcome
from .paths import resolved_layout
from .pr_create_retry import create_pr_with_retry
from .process_utils import (
    is_pid_alive,
    sweep_orphan_processes,
)
from .review_decision import review_decision
from .rework_prompts import _write_rework_prompt
from .state import (
    load_state,
    load_state_locked,
    set_throttled_until,
    state_lock,
)
from .worker import WorkerHealth, WorkerView
from .worktree import (
    WORKTREE_UNSAFE_KINDS,
    SalvagePushResult,
    inspect_worktree_state,
    push_branch,
    resolve_base_branch_name,
    salvage_push_stranded_commits,
    summarize_branch_work,
    worktree_ahead_of_sha,
    worktree_path_for_branch,
)
from .salvage_superseded import check_salvage_superseded, salvage_skip_event_kind
from .write_gate import WriteGate, require_write_gate


# Issue #1106: a rework session that dies at CLI startup (before the worker's
# first tool action) is not a no-op/conflict rework attempt — the cap counters
# should only count sessions that *ran* and produced no useful change.  A
# ``launch_failed`` (process never launched) is always a startup death; a
# ``stalled`` session that died within this threshold is also a startup death
# (CLI error, nonzero exit within seconds, empty diff AND empty transcript).
# The threshold is deliberately generous: a genuine stall (worker ran for
# minutes but got stuck) must NOT be misclassified as a startup death, so the
# bound is set above the worst-case CLI startup time but well below the
# shortest genuine-work session.
#
# The ``runtime_seconds`` passed to ``_is_startup_death`` must be bounded by
# the worker's *actual process runtime* (time from start to death), NOT by the
# elapsed time until the orchestrator's classification pass runs.  Using
# ``WorkerView.runtime_seconds()`` (which is ``now - started_at``) would let
# ordinary polling latency — the gap between when the CLI died and when the
# reaper pass classifies it — silently push a 5-second startup death past the
# 60-second threshold and defeat the exemption.  ``_worker_death_bounded_runtime_seconds``
# derives the runtime from the log file's last-modified time (frozen at death
# for a dead process) instead.
STARTUP_DEATH_THRESHOLD_SECONDS: int = 60


def _is_startup_death(failure_kind: str | None, runtime_seconds: float) -> bool:
    """Classify whether a dead rework session died at CLI startup.

    Returns True when the session never reached the worker's first tool
    action — the cap counters in ``_route_janitor_gate_failure_to_rework``
    must not count these as no-op/conflict rework attempts (issue #1106).

    ``runtime_seconds`` must be the worker's *death-bounded* runtime (time
    from ``started_at`` to the last real log activity / death), as computed
    by ``_worker_death_bounded_runtime_seconds`` — NOT
    ``WorkerView.runtime_seconds()``, which measures elapsed time until
    classification and is polluted by polling latency.
    """
    if failure_kind is None:
        return False
    # ``launch_failed``: the process never launched at all.
    if failure_kind == "launch_failed":
        return True
    # ``stalled`` with a very short runtime: the CLI exited before the
    # worker did any real work (e.g. "Refusing to run in an untrusted
    # workspace").  A longer runtime means the worker genuinely ran and
    # got stuck — that IS a no-op rework attempt the cap should count.
    if failure_kind == "stalled" and runtime_seconds < STARTUP_DEATH_THRESHOLD_SECONDS:
        return True
    return False


def _worker_death_bounded_runtime_seconds(worker: WorkerView) -> float:
    """Return the worker's runtime bounded by actual process death.

    This is the signal ``_is_startup_death`` must use instead of
    ``WorkerView.runtime_seconds()`` (which is ``now - started_at`` and
    measures elapsed time until the *classification pass*, not until death).
    A CLI that dies at 5 seconds but is not classified until 300 seconds
    later must still be recognized as a 5-second startup death, not a
    300-second stall.

    The death-bounded runtime is derived from the log file's last-modified
    time: once the CLI process exits, the log stops being written and its
    mtime freezes at the death moment.  A fresh ``stat()`` of the log file
    is the most accurate signal; the sidecar's recorded ``last_activity_at``
    (updated each pass by ``update_worker_log_stat``) is the fallback when
    the log file is gone.  When neither is available — the CLI never wrote
    anything, e.g. a ``launch_failed`` that still got a PID — the runtime is
    0.0, which is a startup death by construction.
    """
    from datetime import UTC, datetime

    try:
        started_at = datetime.fromisoformat(worker.started_at)
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return 0.0

    # Prefer a fresh stat of the log file — its mtime is frozen at death for
    # a dead process, so this is the tightest death-bounded signal.
    death_ts: float | None = None
    log_stat = worker.log_stat()
    if log_stat is not None:
        death_ts = log_stat.st_mtime
    if death_ts is None and worker.last_activity_at is not None:
        # Fall back to the sidecar's recorded last-activity timestamp.
        from .worker import _iso_to_timestamp

        death_ts = _iso_to_timestamp(worker.last_activity_at)
    if death_ts is None:
        # No log activity was ever recorded — the CLI never wrote anything,
        # which is a startup death by construction (runtime 0.0).
        return 0.0
    return max(0.0, death_ts - started_at.timestamp())


def _session_failed_relabeled_payload(
    *,
    issue_number: int,
    reason: str,
    failure_kind: str | None = None,
    removed_labels: list[str] | tuple[str, ...] = (),
    added_ready: bool = False,
    label_write_ok: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    """Build the payload for a ``session_failed_relabeled`` event.

    Mirrors ``_escalate_issue`` (#750): ``reason`` is keyword-only and
    required, so a relabel event can never be emitted without saying *why*
    it fired. ``failure_kind`` is an optional refinement (the classifier's
    verdict, which may be ``None`` when classification could not determine
    a kind); when present it is recorded alongside ``reason``, but
    ``reason`` is the canonical field a reader filters on.

    Every prior call site spelled "why" under a different key --
    ``reason`` (orphan sweep), ``failure_kind`` only (dead-worker
    no-open-PR), both (phantom live worker), or English prose in
    ``detail`` (reconcile) -- so a query on any one key silently missed
    the rows written by the others. Routing all sites through this
    builder makes the omission unrepresentable: there is no way to call
    it without a ``reason``. Issue #978.
    """
    payload: dict[str, Any] = {
        "issue_number": issue_number,
        "reason": reason,
        "removed_labels": sorted(removed_labels),
        "added_ready": added_ready,
        "label_write_ok": label_write_ok,
    }
    if failure_kind is not None:
        payload["failure_kind"] = failure_kind
    if extra:
        payload.update(extra)
    return payload


def _emit_session_failed_relabeled(
    state: dict[str, Any],
    *,
    issue_number: int,
    reason: str,
    failure_kind: str | None = None,
    removed_labels: list[str] | tuple[str, ...] = (),
    added_ready: bool = False,
    label_write_ok: bool = True,
    state_path: Path | None = None,
    write_gate: WriteGate,
    **extra: Any,
) -> dict[str, Any]:
    """Emit a ``session_failed_relabeled`` event via the shared payload builder.

    Thin wrapper around :func:`_session_failed_relabeled_payload` plus
    :func:`append_event`; see the payload builder's docstring for the
    required-``reason`` invariant (issue #978).

    Issue #1264 (W6 PR3): ``write_gate`` is declared explicitly (not folded
    into ``**extra``) so a caller that forgets it gets ``require_write_gate``'s
    loud ``TypeError`` instead of the gate silently landing in the event
    payload. ``state_path`` is consequently now vestigial (the gate
    auto-binds its own ``state_path``) but kept as a parameter rather than
    removed, the same call PR2 made for the analogous
    ``_merge_on_write_save`` seam (adversarial-review finding F3), to avoid
    churning callers for no behavioral gain.
    """
    write_gate = require_write_gate(write_gate)
    return write_gate.append_event(
        state,
        "session_failed_relabeled",
        _session_failed_relabeled_payload(
            issue_number=issue_number,
            reason=reason,
            failure_kind=failure_kind,
            removed_labels=removed_labels,
            added_ready=added_ready,
            label_write_ok=label_write_ok,
            **extra,
        ),
    )


def _count_live_sessions(sessions_dir: Path, state_file: Path | None = None) -> int:
    """Count the number of currently alive worker sessions across both adapters.

    Reads session sidecar files from both devin-shell and claude-code adapters,
    then checks each record's PID liveness using the adapter-specific liveness
    probe. Returns the total count of sessions with alive PIDs.

    When ``state_file`` is given, this also corroborates the sidecar-based
    count against state.json's own dispatched-issue ``worker_pid``/
    ``worker_process_start_time`` records (issue #343). A sidecar can go
    missing for a still-live process -- a "ghost" -- if a reap lane removes
    it on ambiguous evidence (see ``_classify_dead_sessions_and_update_
    throttle_state``'s corroboration gate) or through any other path that
    strands state.json's dispatch record. Because the governor's dispatch
    cap (``_apply_concurrency_governor``) is built on top of this count, a
    ghost previously made a live process invisible to concurrency accounting
    and let the next pass over-dispatch past the configured cap. state.json's
    worker_pid fields are not touched by sidecar reaping, so any issue still
    recorded as ``dispatched`` whose worker_pid is alive (pid + start-time,
    recycling-safe) but has no corresponding live sidecar is counted here
    too, and printed as a loud reconcile signal rather than silently treated
    as free capacity.
    """
    from .worker import iter_workers

    live_issue_numbers: set[int] = set()
    live_count = 0
    for w in iter_workers(sessions_dir):
        if w.is_alive():
            live_count += 1
            live_issue_numbers.add(w.issue_number)

    if state_file is not None:
        state = load_state_locked(state_file)
        for issue_number_str, entry in state.get("issues", {}).items():
            if not isinstance(entry, dict) or entry.get("status") != "dispatched":
                continue
            try:
                issue_number = int(issue_number_str)
            except (TypeError, ValueError):
                continue
            if issue_number in live_issue_numbers:
                continue
            if _worker_pid_alive(entry):
                live_count += 1
                print(
                    f"[reconcile] issue {issue_number}: worker_pid "
                    f"{entry.get('worker_pid')} is alive with no live session "
                    "sidecar (ghost) -- counting it against the concurrency "
                    "governor instead of treating the slot as free",
                    flush=True,
                )

    return live_count


def _detect_stalled_sessions(
    sessions_dir: Path, config: OrchestratorConfig, *, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Detect stalled sessions (live PID but dead agent) without handling them.

    A session is stalled when its PID is alive but its log file's mtime is
    older than the configured threshold, or the log contains a terminal error
    marker. This is a read-only detection function for status/roll-call.

    ``now`` (issue #828) is the injectable clock, following the convention
    established by PR #827 / ``get_rate_limit_defer_until``: defaults to
    ``datetime.now(UTC)`` when omitted, so production behavior is
    byte-identical.

    Returns a list of {issue, pid, health, terminal_tool, terminal_reason}
    dicts for stalled sessions. ``health`` distinguishes STALLED from DEAD
    (issue #261) so digest callers can surface dead-worker terminal cause
    instead of collapsing everything to "STALLED". ``terminal_tool``/
    ``terminal_reason`` are populated only for DEAD entries with a matching
    post-mortem sidecar (best-effort — absent when extraction found nothing).
    """
    from datetime import UTC, datetime
    from .post_mortem import read_post_mortem
    from .worker import classify_worker_health, iter_workers, real_activity_probe_for

    if not config.watchdog.enabled:
        return []

    stalled_entries: list[dict[str, Any]] = []
    if now is None:
        now = datetime.now(UTC)

    for w in iter_workers(sessions_dir):
        if w.pid is None or w.error is not None:
            continue

        # Issues #280/#301: corroborate against real-session activity for read-only
        # detection too (status/dry-run/digest).
        probe = real_activity_probe_for(w, config, now)
        health = classify_worker_health(w, config, now, probe)

        # Both STALLED and DEAD are considered "stalled" for reporting purposes
        if health in (WorkerHealth.STALLED, WorkerHealth.DEAD):
            entry: dict[str, Any] = {
                "issue": w.issue_number,
                "pid": w.pid,
                "health": health.name,
            }
            if health is WorkerHealth.DEAD:
                post_mortem = read_post_mortem(sessions_dir, w.issue_number)
                if post_mortem is not None:
                    entry["terminal_tool"] = post_mortem.terminal_tool
                    entry["terminal_reason"] = post_mortem.terminal_reason
            stalled_entries.append(entry)

    return stalled_entries


# NOTE: the former `_kill_orphan_pid` free function that lived here was
# hoisted to `process_utils.kill_orphan_pid` verbatim (issue #1264, W6 PR3,
# R6a) so that `write_gate.py` can wrap it as `WriteGate.kill_process`
# without importing this module. The two call sites below
# (`_detect_and_handle_stalled_sessions`) are unconditional and out of this
# PR's scope -- see issue #1311's sibling filing for that function's own
# dry-run leak -- so they call the hoisted primitive raw, unchanged in
# behavior. The one in-scope call site (`_sweep_orphan_processes_for_dead_sessions`)
# now goes through `write_gate.kill_process` instead.


def _detect_and_handle_stalled_sessions(
    sessions_dir: Path,
    state_file: Path,
    config: OrchestratorConfig,
    *,
    write_gate: WriteGate,
    now: datetime | None = None,
) -> list[dict[str, int]]:
    """Detect stalled sessions (live PID but dead agent) and handle them.

    A session is stalled when its PID is alive but its log file's mtime is
    older than the configured threshold, or the log contains a terminal error
    marker. On detection, the process tree is killed and the sidecar is
    classified via the log-tail-first helper (``update_session_record_with_
    failure_classification`` / ``update_worker_record_with_failure_classification``),
    falling back to failure_kind "stalled" only when the log shows no provider
    throttle signature. This function runs before the dead-session lane in the
    loop() pass order, so it must apply the same classify-then-fallback
    treatment itself — otherwise a worker that actually died on a provider
    rate limit gets permanently mislabeled "stalled" before the dead-session
    lane ever gets a chance to classify it, and ``throttled_until`` never gets
    set (issue #246).

    When classification detects a throttle signature, ``throttled_until`` is
    persisted to state.json (same as the dead-session lane) so the next
    dispatch pass defers instead of relaunching into the same window. A
    session_stalled event is logged with the resolved failure_kind.

    Issue #873: the reap event's *kind* now follows the worker's health.
    ``WorkerHealth.STALLED`` (a live process that stopped progressing) keeps
    the error-level ``session_stalled``; ``WorkerHealth.DEAD`` (the process is
    already gone — equally the normal end of a worker that finished and
    exited) emits the warning-level ``session_exited``. The handling is
    deliberately still shared; only the classification splits. Both payloads
    carry ``worker_health`` so the distinction is recorded rather than
    inferred from an empty ``killed_pids``.

    ``now`` (issue #828) is the injectable clock for this sweep, following
    the convention established by PR #827. It is sampled once here and
    threaded down to ``real_activity_probe_for``, ``classify_worker_health``,
    ``get_rate_limit_defer_until``, and ``classify_and_record`` so a single
    pass never straddles two different clock samples. Defaults to
    ``datetime.now(UTC)`` when omitted, so production behavior is
    byte-identical.

    Returns a list of {issue, pid} dicts for stalled sessions (for exclusion from
    dispatch in the same pass).

    Issue #1325: all process kills (``kill_process_tree``,
    ``kill_orphan_pid``), state/event writes (``append_event``,
    ``save_state``), and sidecar writes (``update_worker_log_stat``, the
    budget-exceeded sidecar write, ``classify_and_record``,
    ``update_session_record_with_failure_classification`` /
    ``update_worker_record_with_failure_classification``) are gated on
    ``write_gate.dry_run`` so a ``dry_run=True`` gate suppresses them
    exactly as ``--dry-run`` promises. The sidecar-writing helpers
    themselves do not accept a ``write_gate``/``dry_run`` parameter; they
    are gated at the call site instead, keeping the change scoped to this
    function (the subject of #1325) and avoiding a caller sweep across
    the dead-session lane and other consumers that intentionally write
    sidecars unconditionally. The ``write_gate`` parameter follows the
    Convention B explicit-threading pattern (``require_write_gate()``)
    already used by ``_sweep_orphan_processes_for_dead_sessions`` and the
    dead-session lane.
    """
    write_gate = require_write_gate(write_gate)
    from .claude_code import update_worker_record_with_failure_classification
    from .devin_shell import (
        get_rate_limit_defer_until,
        update_session_record_with_failure_classification,
    )
    from .post_mortem import classify_and_record
    from .worker import (
        _next_inconclusive_probe_deferred_count,
        _api_session_over_budget,
        classify_worker_health,
        iter_workers,
        real_activity_probe_for,
        update_worker_log_stat,
    )

    if not config.watchdog.enabled:
        return []

    stalled_entries: list[dict[str, int]] = []
    if now is None:
        now = datetime.now(UTC)

    def _parse_defer_until(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed
        except (ValueError, TypeError):
            return None

    def _is_deferred(w: Any) -> bool:
        """Return True if the worker has a stored defer deadline that has not passed."""
        defer_until = _parse_defer_until(w.rate_limit_defer_until)
        if defer_until is None:
            return False
        return now < defer_until

    for w in iter_workers(sessions_dir):
        if w.pid is None or w.error is not None:
            continue

        # Update log stat fields for progress tracking. This also clears any
        # stale rate-limit defer deadline when the log has resumed growing.
        # Issue #1325: gated on ``write_gate.dry_run`` so the sidecar is not
        # mutated under ``--dry-run``. Detection (``classify_worker_health``,
        # ``real_activity_probe_for``) operates on the pre-fetched
        # ``WorkerView`` from ``iter_workers`` and does not re-read the
        # sidecar, so skipping this write does not affect this pass's
        # classification — it only suppresses the sidecar mutation for the
        # next pass, which is exactly what ``--dry-run`` promises.
        if not write_gate.dry_run:
            update_worker_log_stat(sessions_dir, w)

        # Issues #280/#301: corroborate sidecar mtime against real-session activity
        # (sessions.db message_nodes, per-PID Devin log mtime, and Claude Code
        # events.jsonl) before deciding whether to kill the worker.
        probe = real_activity_probe_for(w, config, now)
        health = classify_worker_health(w, config, now, probe)

        # Issue #338: persist Signal-1's inconclusive-probe deferral counter so the
        # escalation cap is tracked across passes. This lane is the sole writer of
        # this counter for a not-alive worker (issue #343 Finding 2): it is the
        # only lane guaranteed to run at least once whenever the watchdog is
        # enabled -- dispatch()/dispatch_rework() each call this lane standalone
        # (no dead-session lane in the same call), and loop() always runs this
        # lane immediately before the dead-session lane
        # (_classify_dead_sessions_and_update_throttle_state). The dead lane
        # deliberately does NOT also persist this counter for a not-alive worker
        # when the watchdog is enabled -- see the comment there -- so it is
        # written at most once per worker per pass instead of twice (0->1 here,
        # then re-read and ->2 there), which halved the effective deferral grace
        # period and was the very mechanism that opened Finding 1's pass-2
        # phantom-sidecar window.
        new_count = _next_inconclusive_probe_deferred_count(w, probe, health)
        # Issue #1325: gated on ``write_gate.dry_run`` so the sidecar is not
        # mutated under ``--dry-run``. The counter is a persisted cross-pass
        # escalation cap; under dry-run it must not advance on disk.
        if not write_gate.dry_run:
            update_worker_log_stat(sessions_dir, w, inconclusive_probe_deferred_count=new_count)

        # Issue #484: in-flight api per-session budget kill. Independent of the
        # STALLED/DEAD classification below — an api worker over its
        # ``max_usd_per_session`` cap is killed immediately and sidecar-marked
        # ``budget_exceeded``. The killed session then flows through the
        # EXISTING dead-worker reconciliation on the next pass (with-PR ->
        # review/rework; without-PR -> re-dispatch via the default adapter,
        # whose preflight naturally decides api-again vs fallback). When the cap is
        # 0/unset the check is entirely dormant. Non-api workers are never
        # budget-evaluated. The kill uses the shared ``kill_process_tree``
        # helper (no-console-window discipline on Windows, full process tree
        # reaped) — not reimplemented here. Issue #1325: both the tree kill
        # and the orphan kill now route through ``write_gate`` so
        # ``dry_run=True`` suppresses them.
        if w.adapter_kind == "api" and _api_session_over_budget(w, config):
            killed_pids = write_gate.kill_process_tree(w.pid, w.process_start_time)
            orphan_pids_budget: list[int] = []
            orphan_processes = sweep_orphan_processes(w.worktree_path)
            if orphan_processes:
                for orphan in orphan_processes:
                    write_gate.kill_process(orphan["pid"])
                    killed_pids.append(orphan["pid"])
                orphan_pids_budget = [o["pid"] for o in orphan_processes]

            # Set failure_kind="budget_exceeded" on the sidecar via the shared
            # atomic-write helper. Written directly (not through
            # update_worker_record_with_failure_classification) so the
            # budget-exceeded verdict is not overridden by a coincidental
            # throttle/auth log-tail match. The dead-session lane's
            # classification call then short-circuits on the already-set
            # failure_kind. Issue #1325: gated on ``write_gate.dry_run`` so
            # the sidecar is not mutated under ``--dry-run``.
            from .claude_code import _sidecar_path as _api_sidecar_path
            from .claude_code import _write_json_atomic as _api_write_json_atomic

            api_sidecar = _api_sidecar_path(sessions_dir, w.issue_number, "api")
            if not write_gate.dry_run:
                try:
                    with api_sidecar.open("r", encoding="utf-8") as handle:
                        api_payload = json.load(handle)
                    if isinstance(api_payload, dict):
                        api_payload["failure_kind"] = "budget_exceeded"
                        _api_write_json_atomic(api_sidecar, api_payload)
                except (OSError, json.JSONDecodeError):
                    pass

            with state_lock(state_file):
                state = load_state(state_file)
                state = write_gate.append_event(
                    state,
                    "session_budget_exceeded",
                    {
                        "issue_number": w.issue_number,
                        "pid": w.pid,
                        "process_start_time": w.process_start_time,
                        "killed_pids": killed_pids,
                        "orphan_pids": orphan_pids_budget if orphan_pids_budget else None,
                        "provider": w.provider,
                    },
                )
                write_gate.save_state(state)

            stalled_entries.append({"issue": w.issue_number, "pid": w.pid})
            continue

        # If a stalled-looking worker is still within a previously stored rate-limit
        # defer window, skip it. The deadline is re-derived from the sidecar each pass.
        if health == WorkerHealth.STALLED and _is_deferred(w):
            continue

        # Both STALLED and DEAD are considered "stalled" for handling purposes
        if health in (WorkerHealth.STALLED, WorkerHealth.DEAD):
            # Issue #247: before killing a stalled-looking worker, check the log tail
            # for a rate-limit signature. If found and we are not already in a defer
            # window, record a defer deadline and skip the kill this pass.
            if health == WorkerHealth.STALLED and config.watchdog.rate_limit_defer_enabled:
                if not _is_deferred(w) and w.rate_limit_defer_until is None:
                    defer_until = get_rate_limit_defer_until(
                        Path(w.log_path),
                        config.watchdog.rate_limit_defer_slack_minutes,
                        now,
                        config.runtime.throttle_error_markers,
                        config.runtime.throttle_resume_margin_s,
                    )
                    if defer_until is not None:
                        # Issue #1325: gated on ``write_gate.dry_run`` so the
                        # sidecar is not mutated under ``--dry-run``.
                        if not write_gate.dry_run:
                            update_worker_log_stat(
                                sessions_dir, w, rate_limit_defer_until=defer_until
                            )
                        with state_lock(state_file):
                            state = load_state(state_file)
                            state = set_throttled_until(
                                state,
                                defer_until,
                                reason="rate_limited",
                                adapter_kind=w.adapter_kind,
                            )
                            state = write_gate.append_event(
                                state,
                                "session_rate_limit_deferred",
                                {
                                    "issue_number": w.issue_number,
                                    "pid": w.pid,
                                    "defer_until": defer_until,
                                },
                            )
                            write_gate.save_state(state)
                        continue

            # Kill the process tree (with start-time verification to prevent PID recycling).
            # Issue #1325: routed through ``write_gate.kill_process_tree`` so
            # ``dry_run=True`` suppresses the kill (returns ``[]``).
            killed_pids = write_gate.kill_process_tree(w.pid, w.process_start_time)

            # Sweep for orphan processes that survived the tree kill (Windows-only)
            # This catches detached/daemonized processes (e.g., nohup-style background processes)
            orphan_pids: list[int] = []
            orphan_processes = sweep_orphan_processes(w.worktree_path)
            if orphan_processes:
                # Kill detected orphans to prevent them from running rejected code.
                # Issue #1325: routed through ``write_gate.kill_process`` so
                # ``dry_run=True`` suppresses the kill.
                for orphan in orphan_processes:
                    write_gate.kill_process(orphan["pid"])
                    killed_pids.append(orphan["pid"])
                orphan_pids = [o["pid"] for o in orphan_processes]

            # Post-mortem extraction (issue #261): reads the Devin CLI's own
            # session store for a terminal-tool diagnosis (esp. decision:block
            # push-gate hooks) independent of the log tail. Runs BEFORE the
            # log-tail classification below — when it detects a block, it
            # writes failure_kind="worker_blocked" directly into the sidecar,
            # which makes the classification call below a no-op via its
            # existing "skip if already classified" short-circuit. Best-effort
            # and read-only: any DB problem degrades to extraction_error and
            # this never changes what happens next.
            # Issue #1325: gated on ``write_gate.dry_run`` so the sidecar is
            # not mutated under ``--dry-run``. The return value is discarded
            # at this call site, so suppressing the call has no downstream
            # effect on this pass's event payload (which itself is suppressed
            # under dry-run via ``write_gate.append_event``).
            if not write_gate.dry_run:
                classify_and_record(sessions_dir, config, w, now=now)

            # Classify the sidecar (adapter-specific dispatch): log-tail
            # classification runs first, falling back to failure_kind "stalled"
            # only when the log shows no provider throttle signature.
            # Issue #1325: gated on ``write_gate.dry_run`` so the sidecar is
            # not mutated under ``--dry-run``. Under dry-run the return values
            # stay ``None``, which means no ``throttled_until`` is persisted
            # (``write_gate.save_state`` is already a no-op) and the event
            # payload (itself suppressed via ``write_gate.append_event``)
            # would carry ``failure_kind=None`` — consistent with "nothing
            # happened," which is exactly what ``--dry-run`` promises.
            resolved_failure_kind: str | None = None
            throttled_until: str | None = None
            if not write_gate.dry_run:
                if w.adapter_kind == "devin":
                    resolved_failure_kind, throttled_until = (
                        update_session_record_with_failure_classification(
                            sessions_dir,
                            w.issue_number,
                            fallback_kind="stalled",
                            config=config,
                        )
                    )
                elif w.adapter_kind == "claude-code":
                    resolved_failure_kind, throttled_until = (
                        update_worker_record_with_failure_classification(
                            sessions_dir,
                            w.issue_number,
                            fallback_kind="stalled",
                            config=config,
                        )
                    )
                elif w.adapter_kind == "api":
                    # api sidecars share the claude-code classification helper but
                    # land as issue-<n>.api.json and get provider-auth classification
                    # (issue #484). adapter_kind="api" selects both the sidecar
                    # suffix and the auth-pattern check inside _classify_session_failure.
                    resolved_failure_kind, throttled_until = (
                        update_worker_record_with_failure_classification(
                            sessions_dir,
                            w.issue_number,
                            fallback_kind="stalled",
                            config=config,
                            adapter_kind="api",
                        )
                    )

            if resolved_failure_kind and throttled_until:
                # A throttle signature was found in the log tail even though
                # the watchdog reaped this worker for stalling — persist the
                # cooldown so the next dispatch pass defers instead of
                # relaunching into the same provider rate limit/quota window.
                with state_lock(state_file):
                    state = load_state(state_file)
                    state = set_throttled_until(
                        state,
                        throttled_until,
                        reason=resolved_failure_kind,
                        adapter_kind=w.adapter_kind,
                    )
                    write_gate.save_state(state)

            # Log the event
            log_path = Path(w.log_path)
            last_log_line = None
            try:
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                lines = log_text.splitlines()
                if lines:
                    last_log_line = lines[-1].strip()
            except OSError:
                pass

            probe_payload = (
                probe.to_payload()
                if probe is not None
                else {
                    "sources": [],
                    "latest_timestamp": None,
                    "latest_source": "probe unavailable",
                }
            )

            # Issue #873: STALLED and DEAD share this handling path, and that
            # sharing is deliberate — the cleanup (reap the tree, free the
            # slot) is identical for both. What they must NOT share is a
            # *classification*. STALLED is a live process that stopped making
            # progress: a genuine fault. DEAD means the process is already
            # gone, which is equally the normal terminal state of every worker
            # that finished its work and exited. Emitting one error-level kind
            # for both filed successful completions as faults in the
            # error-level stream that #864/#866 added the first consumer for
            # (scripts/heartbeat_check.check_error_events).
            #
            # The reporting lane (``_detect_stalled_sessions``) already kept
            # ``health.name`` in its entry; only this handling lane discarded
            # the distinction at the point of emission. Both the event kind and
            # the new ``worker_health`` payload field now carry it, so the
            # split is recorded rather than inferred from ``killed_pids``.
            #
            # ``session_exited`` is deliberately not named
            # ``session_completed``: process liveness proves the worker is
            # gone, not that it succeeded. For the same reason it is a warning
            # rather than info — this change does not establish the
            # clean-exit-vs-crash split, and info would encode a claim the
            # evidence does not support.
            event_kind = "session_stalled" if health is WorkerHealth.STALLED else "session_exited"

            with state_lock(state_file):
                state = load_state(state_file)
                state = write_gate.append_event(
                    state,
                    event_kind,
                    {
                        "issue_number": w.issue_number,
                        "pid": w.pid,
                        "process_start_time": w.process_start_time,
                        "worker_health": health.name,
                        "log_mtime": str(datetime.fromtimestamp(log_path.stat().st_mtime, tz=UTC)),
                        "last_log_line": last_log_line,
                        "killed_pids": killed_pids,
                        "orphan_pids": orphan_pids if orphan_pids else None,
                        "failure_kind": resolved_failure_kind,
                        "activity_sources": probe_payload.get("sources", []),
                        "latest_real_activity_at": probe_payload.get("latest_timestamp"),
                        "latest_real_activity_source": probe_payload.get("latest_source"),
                    },
                )
                write_gate.save_state(state)

            stalled_entries.append({"issue": w.issue_number, "pid": w.pid})

    return stalled_entries


def _worker_pid_alive(entry: dict[str, Any]) -> bool:
    """Check if a worker PID from state.json is alive, with start-time verification.

    This helper deduplicates the PID liveness check used across dispatch and
    orphaned worker detection. It checks both PID liveness and process identity
    via start time to detect PID recycling.

    Args:
        entry: A state.json issue entry containing worker_pid and optionally
               worker_process_start_time.

    Returns:
        True if the worker PID is alive and the start time matches (if available),
        False otherwise.
    """
    worker_pid = entry.get("worker_pid")
    if worker_pid is None:
        return False

    return is_pid_alive(worker_pid, entry.get("worker_process_start_time"))


def _orphan_head_fingerprint(remote_sha: str | None, local_sha: str | None) -> str:
    """Combine remote and local branch head SHAs into a single progress fingerprint.

    A change in either SHA indicates progress -- a remote push or a local
    (possibly stranded) commit. Used by the orphan-sweep redispatch cap
    (issue #1243) to distinguish a no-progress death loop (unchanged fingerprint
    across attempts) from a moving head that is the salvage path's job.
    """
    return f"{remote_sha or 'none'}:{local_sha or 'none'}"


# Issue #1153: minimum number of zero-artifact attempts (all ``ahead_of_main
# == 0``) in a post-mortem sidecar before the orphan sweep escalates instead
# of relabeling to ``automated-ready`` for another redispatch. The first
# zero-artifact attempt is the initial dispatch; the second is the first
# redispatch. Escalating before the *second* redispatch (the third attempt
# overall) means the threshold is 2: two attempts have already produced zero
# artifacts, so a third would almost certainly do the same.
_ZERO_ARTIFACT_ESCALATION_THRESHOLD = 2


def _is_zero_artifact_dispatch_loop(sessions_dir: Path, issue_number: int) -> bool:
    """Return True when prior dispatch attempts all produced zero artifacts.

    Issue #1153: an issue whose post-mortem sidecar records ``>= 2`` attempts
    where *every* attempt's ``ahead_of_main`` is ``0`` is in a zero-artifact
    dispatch loop -- each worker session ran, produced no commits ahead of
    the base ref, and was swept as a dead worker with no open PR. The
    post-mortem file already contains exactly the signal needed
    (``attempts[].ahead_of_main == 0`` repeated); this helper reads it so the
    orphan sweep can escalate to ``agent:human-needed`` instead of relabeling
    to ``automated-ready`` for yet another fruitless redispatch.

    Returns ``False`` when there is no sidecar, fewer than the threshold
    number of attempts, any attempt has a non-zero ``ahead_of_main``, or any
    attempt's ``ahead_of_main`` is ``None`` (unknown -- cannot confirm
    zero-artifact, so do not escalate on ambiguous evidence).
    """
    from .post_mortem import read_post_mortem

    record = read_post_mortem(sessions_dir, issue_number)
    if record is None:
        return False
    if len(record.attempts) < _ZERO_ARTIFACT_ESCALATION_THRESHOLD:
        return False
    # Every attempt must have a confirmed ahead_of_main == 0. An attempt
    # with ahead_of_main == None is ambiguous (the count could not be
    # computed) -- do not escalate on ambiguous evidence.
    return all(attempt.ahead_of_main == 0 for attempt in record.attempts)


def _sweep_orphan_processes_for_dead_sessions(
    sessions_dir: Path,
    state_file: Path,
    config: OrchestratorConfig,
    *,
    write_gate: WriteGate,
) -> None:
    """Sweep for orphan processes in worktrees of dead sessions.

    This is called from the production loop to detect and clean up orphaned
    processes that survived session kills (e.g., detached/daemonized processes
    like nohup-style background processes). This addresses issue #139.

    On Windows: Uses PowerShell Get-CimInstance Win32_Process to find processes
    whose CommandLine references the worktree path of dead sessions.
    On POSIX: Not implemented (returns empty list).

    Detected orphans are killed automatically and logged to state.json.

    Issue #1264 (W6 PR3): this is one of the three unconditional
    ``_loop_body`` call sites named by issue #1311's dry-run leak. Both the
    orphan kill and the event/state write below now go through
    ``write_gate`` -- the kill via the 6th gate method, ``kill_process``
    (R6a: wraps the ``_kill_orphan_pid`` primitive hoisted to
    ``process_utils.kill_orphan_pid``).
    """
    write_gate = require_write_gate(write_gate)
    from .devin_shell import is_session_alive, read_session_records
    from .claude_code import is_worker_alive, read_worker_records

    # Only run on Windows where the issue occurs
    if os.name != "nt":
        return

    # Collect worktree paths of dead sessions
    dead_worktree_paths: set[str] = set()

    # Check devin-shell sessions
    for record in read_session_records(sessions_dir):
        if record.pid is None or record.error is not None:
            continue
        if not is_session_alive(record):
            dead_worktree_paths.add(record.worktree_path)

    # Check claude-code sessions
    for record in read_worker_records(sessions_dir):
        if record.pid is None or record.error is not None:
            continue
        if not is_worker_alive(record):
            dead_worktree_paths.add(record.worktree_path)

    # Sweep for orphans in each dead worktree
    for worktree_path in dead_worktree_paths:
        orphan_processes = sweep_orphan_processes(worktree_path)
        if orphan_processes:
            # Kill detected orphans
            killed_orphans: list[int] = []
            for orphan in orphan_processes:
                write_gate.kill_process(orphan["pid"])
                killed_orphans.append(orphan["pid"])

            # Log the event with image/cmdline of each killed process so the
            # respawn source in dead worktrees can be identified and shut off.
            with state_lock(state_file):
                state = load_state(state_file)
                state = write_gate.append_event(
                    state,
                    "orphan_processes_killed",
                    {
                        "worktree_path": worktree_path,
                        "orphan_pids": [o["pid"] for o in orphan_processes],
                        "killed_orphans": killed_orphans,
                        "orphan_processes": [
                            {
                                "pid": o["pid"],
                                "name": o.get("name"),
                                "command_line": o.get("command_line"),
                            }
                            for o in orphan_processes
                        ],
                    },
                )
                write_gate.save_state(state)


def _log_worker_census(sessions_dir: Path) -> None:
    """Log one INFO line per loop pass listing every currently-alive worker.

    Issue #646: the launch-time log in claude_code.py/devin_shell.py answers
    "what cap did this worker launch with", but says nothing about how many
    are running *right now* -- the question a box-saturation incident needs
    answered ("how many suites were running at 11:33, from which worktrees,
    at what cap"). This sweep answers it directly from log content alone,
    with no process forensics required.

    Deliberately read-only (no state mutation): it carries none of the
    fragile sole-writer invariants _detect_and_handle_stalled_sessions/
    _sweep_orphan_processes_for_dead_sessions must protect. It also only
    ever iterates *alive* records, so — unlike a dead/exit-transition sweep —
    it has no "months of accumulated stale sidecar" flooding problem even if
    old sidecars are never pruned from sessions_dir.

    Called from the top of ``dispatch()`` (see its docstring) -- the one
    chokepoint every dispatch path funnels through, standalone (`work`/`fleet
    work`) or supervised (`loop()` -> `_loop_body()` -> `dispatch()`) -- so it
    runs unconditionally regardless of how long the orchestrator process
    itself lives: a one-shot ``charlie fleet work`` CLI invocation logs
    exactly one census line before exiting; a long-lived ``charlie fleet
    supervise`` logs one per pass. Both answer the diagnostic question above
    from log content alone -- no need to correlate against a live process
    list.
    """
    import logging

    logger = logging.getLogger(__name__)

    from .claude_code import is_worker_alive, read_worker_records
    from .devin_shell import is_session_alive, read_session_records

    now = datetime.now(UTC)

    def _age_seconds(started_at: str) -> int | None:
        try:
            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        except (ValueError, TypeError, AttributeError):
            return None
        return int((now - started).total_seconds())

    entries: list[str] = []
    # adapter_kind=None also covers the "api" adapter, which delegates to
    # claude_code.launch_claude_worker under the hood and shares its sidecar
    # schema (and therefore xdist_cap/is_worker_alive) unchanged.
    for record in read_worker_records(sessions_dir, adapter_kind=None):
        if record.pid is None or record.error is not None or not is_worker_alive(record):
            continue
        entries.append(
            f"(adapter={record.adapter_kind} issue={record.issue_number} "
            f"worktree={record.worktree_path} pid={record.pid} "
            f"cap={record.xdist_cap} age_s={_age_seconds(record.started_at)})"
        )
    for record in read_session_records(sessions_dir):
        if record.pid is None or record.error is not None or not is_session_alive(record):
            continue
        entries.append(
            f"(adapter=devin-shell issue={record.issue_number} "
            f"worktree={record.worktree_path} pid={record.pid} "
            f"cap={record.xdist_cap} age_s={_age_seconds(record.started_at)})"
        )

    logger.info("worker census: n_alive=%d %s", len(entries), " ".join(entries) or "[]")


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

        # Issue #315 finding 2: same window-filtered redispatch_at bookkeeping
        # the sibling lanes use (~line 950-961, ~4186-4194), so the cap below
        # is actually consulted instead of silently never growing.
        redispatch_at = _windowed_redispatch_at(
            entry, window_minutes=config.watchdog.redispatch_window_minutes
        ) + [datetime.now(UTC).isoformat().replace("+00:00", "Z")]

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
        # Record this death in worker_death_at (parallel to the orphan sweep
        # at ~line 4245), and separate the death count from the no-op count
        # in the cap check below.  A death-loop escalates with
        # worker_death_loop (triage: "check the worktree for stranded work")
        # instead of redispatch_cap_exceeded (triage: "worker is spinning").
        worker_death_at = _windowed_worker_death_at(
            entry, window_minutes=config.watchdog.redispatch_window_minutes
        )
        if not immediate_escalation:
            worker_death_at = worker_death_at + [
                datetime.now(UTC).isoformat().replace("+00:00", "Z")
            ]

        no_op_count = max(0, len(redispatch_at) - len(worker_death_at))
        death_count = len(worker_death_at)
        death_loop = not immediate_escalation and death_count > config.watchdog.max_auto_redispatch
        no_op_loop = (
            not immediate_escalation
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

    status_rollup = pr.get("statusCheckRollup")
    if status_rollup:
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
        ) + [datetime.now(UTC).isoformat().replace("+00:00", "Z")]

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


def _classify_dead_sessions_and_update_throttle_state(
    sessions_dir: Path,
    state_file: Path,
    gh: GitHubLike,
    config: OrchestratorConfig,
    *,
    write_gate: WriteGate,
    persist_inconclusive_probe_counter: bool = True,
    now: datetime | None = None,
    fleet_dir_override: str | None = None,
) -> list[dict[str, Any]]:
    """Check for dead sessions, classify their failures, and update throttle state.

    This is called from the production loop to detect provider throttling
    from worker deaths and set the cooldown window in state.json.

    Also reconciles labels for dead sessions with no open PR (issue #118):
    a dead worker with no open PR is recoverable and should be relabeled
    as dispatchable (remove active labels, ensure ready label present).

    Issue #252: if a dead worker has a clean worktree with unpushed commits
    (completed-but-unpublished), push the branch, create a PR, and move the
    issue to ``pr_open`` instead of re-dispatching.

    Issue #266: launch-failure sidecars (pid=None, error set) are terminal by
    construction and are reaped immediately, reported in the returned list.

    Issue #295: dead/launch-failed rework sessions with an open PR and a
    request_changes verdict (or a rework prompt on disk) are restored to
    ``rework_requested`` so ``dispatch_rework`` can re-select them.

    ``persist_inconclusive_probe_counter`` (issue #343 Finding 2): controls
    whether this lane persists Signal-1's inconclusive-probe deferral counter
    for a not-alive, pid-bearing worker. Defaults to True so this function
    remains fully self-sufficient when called on its own (as every existing
    unit test does, and as any future standalone caller would expect).
    ``loop()`` is the one caller that always runs the sibling stall lane
    (``_detect_and_handle_stalled_sessions``, the sole writer of this counter
    for an ALIVE-but-stalled worker, and -- unconditionally, regardless of
    liveness -- the first lane to see every worker each pass) immediately
    before this one, in the same pass; it passes False there so this lane
    does not ALSO increment the same counter on top of what the stall lane
    just wrote a moment earlier -- that double counting (0->1 in the stall
    lane, then re-read and ->2 here) halved the effective deferral grace
    period and was the very mechanism that opened Finding 1's pass-2
    phantom-sidecar window. classify_worker_health's own cap-check always
    reads whatever value is currently on the sidecar, regardless of which
    lane -- or how many passes ago -- last wrote it, so suppressing the
    write here never affects the DEAD-vs-deferred decision made below,
    only which lane's write ends up on disk for a given pass.

    ``now`` (issue #822) is the injectable clock for this entire pass: it
    seeds ``now_for_health`` (worker-health/probe timing) and is forwarded
    to every throttle classification call below, so a single instant is used
    consistently for the whole pass instead of each call independently
    racing the wall clock. Defaults to ``datetime.now(UTC)`` when omitted,
    so production behavior is byte-identical; tests can freeze it and assert
    exact equality on the resulting ``throttled_until`` instead of a
    wall-clock-tolerance proximity check.
    """
    write_gate = require_write_gate(write_gate)
    from .claude_code import update_worker_record_with_failure_classification
    from .devin_shell import update_session_record_with_failure_classification
    from .post_mortem import classify_and_record
    from .state import load_state, set_throttled_until, state_lock
    from .worker import (
        is_worker_confirmed_dead,
        iter_workers,
    )
    from .worktree import WorktreeState

    now_for_health = now if now is not None else datetime.now(UTC)

    # Fetch open PRs for the "no open PR" guard
    prs = gh.pr_list()
    open_prs_by_issue: dict[int, list[dict[str, Any]]] = {}
    # Issue #1229: validate branch-name-derived issue numbers against the
    # open-issue set so a stale branch name (e.g. agent/issue-709-… left over
    # from a merged PR #709, reused by an unrelated issue-less PR) cannot
    # populate open_prs_by_issue[<wrong n>]. That map feeds the
    # escalation/salvage-skip guard below (``w.issue_number not in
    # open_prs_by_issue``): a stale binding to a dead worker's issue number
    # would make the guard see a phantom open PR for that issue and skip
    # escalation/salvage for a session that has no real open PR — the same
    # "act on the wrong subject" failure class as the rework-episode
    # collision the issue was filed for.
    branch_validator = build_branch_issue_validator(gh)
    for pr in prs:
        pr_state = str(pr.get("state") or "").upper()
        if pr_state != "OPEN":
            continue
        issue_number = linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=config.dispatch.branch_prefix,
            branch_issue_validator=branch_validator,
        )
        if issue_number is not None:
            open_prs_by_issue.setdefault(issue_number, []).append(pr)

    repo_root = getattr(gh, "repo_root", None)
    # Issue #1244: pre-compute fleet info for the cross-repo scope tripwire.
    # The managed-repo set is derived from the fleet registry, never a
    # hardcoded list.  Computed once before the worker loop because the fleet
    # registry is read from disk and gh.name_with_owner() is a network call.
    sweep_fleet_repos = managed_repo_names(fleet_dir_override)
    sweep_dispatching_repo_name = (
        _dispatching_repo_name(gh, repo_root) if repo_root is not None else ""
    )

    reaped: list[dict[str, Any]] = []

    for w in iter_workers(sessions_dir):
        if w.pid is None and w.error is not None:
            # Launch-failure sidecar: terminal by construction (issue #266).
            # The process never launched, so it can never transition to live.
            failure_kind = "launch_failed"
            throttled_until = None
            if w.adapter_kind == "devin":
                failure_kind, throttled_until = update_session_record_with_failure_classification(
                    sessions_dir,
                    w.issue_number,
                    fallback_kind=failure_kind,
                    config=config,
                    now=now_for_health,
                )
            elif w.adapter_kind == "claude-code":
                failure_kind, throttled_until = update_worker_record_with_failure_classification(
                    sessions_dir,
                    w.issue_number,
                    fallback_kind=failure_kind,
                    config=config,
                    now=now_for_health,
                )
            elif w.adapter_kind == "api":
                failure_kind, throttled_until = update_worker_record_with_failure_classification(
                    sessions_dir,
                    w.issue_number,
                    fallback_kind=failure_kind,
                    config=config,
                    adapter_kind="api",
                    now=now_for_health,
                )
            if failure_kind and throttled_until:
                # A throttle-caused launch failure must persist its window just
                # like the dead-session branch below — otherwise the governor
                # relaunches straight into the same throttled provider.
                with state_lock(state_file):
                    state = load_state(state_file)
                    state = set_throttled_until(
                        state,
                        throttled_until,
                        reason=failure_kind,
                        adapter_kind=w.adapter_kind,
                    )
                    write_gate.save_state(state)

            if (
                failure_kind in DETERMINISTIC_ESCALATION_FAILURE_KINDS
                or failure_kind in DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS
            ) and w.issue_number not in open_prs_by_issue:
                try:
                    issue = gh.issue_view(w.issue_number)
                except Exception:
                    issue = None
                issue_labels = label_names(issue) if issue else set()
                active_labels = issue_labels & config.labels.active

                # Issue #1130: before escalating a ``worktree_unsafe`` launch
                # failure, attempt salvage. A ``worktree_unsafe`` failure means
                # redispatch found the worktree holding unpushed commits — that
                # is exactly the work salvage exists to recover. Escalating to
                # a human without attempting the cheap safe action (push the
                # branch + open a PR) inverts the priority: salvage-the-commit
                # first, human adjudication only when salvage fails.
                # Issue #807: ``worktree_unsafe`` is split at detection time
                # into ``worktree_unsafe_shim_dirt`` and
                # ``worktree_unsafe_local_commits``; both are covered by
                # ``WORKTREE_UNSAFE_KINDS``. The ``ahead_count > 0`` gate below
                # naturally filters shim dirt (uncommitted modifications, no
                # commits ahead) so salvage only fires for genuine local
                # commits — the case it was designed for.
                salvaged_from_unsafe = False
                if (
                    failure_kind in WORKTREE_UNSAFE_KINDS
                    and repo_root is not None
                    and w.branch
                    and active_labels
                ):
                    worktrees_dir = resolved_layout(config, repo_root).worktrees
                    wt_path = worktree_path_for_branch(repo_root, w.branch, worktrees_dir)
                    unsafe_inspection = inspect_worktree_state(
                        wt_path,
                        config.dispatch.base_ref,
                        config.dispatch.injected_paths,
                        config.dispatch.materialize_dirs,
                    )
                    if unsafe_inspection.ahead_count > 0:
                        salvaged_from_unsafe, _ = _attempt_salvage(
                            gh=gh,
                            config=config,
                            repo_root=repo_root,
                            worktree_path=wt_path if wt_path.is_dir() else repo_root,
                            branch=w.branch,
                            base_ref=unsafe_inspection.resolved_base_ref
                            or config.dispatch.base_ref,
                            issue_number=w.issue_number,
                            active_labels=active_labels,
                            issue_labels=issue_labels,
                            state_file=state_file,
                            failure_kind=failure_kind,
                            issue_title=issue.get("title") if issue else None,
                            # Issue #1241: pass the live issue so the shared
                            # supersession check's closed-issue branch fires
                            # here too -- without it the check would have to
                            # re-fetch via ``gh.issue_view`` (an extra round
                            # trip) and, worse, the caller already has the
                            # freshest snapshot.
                            issue=issue,
                            write_gate=write_gate,
                        )

                if not salvaged_from_unsafe:
                    with state_lock(state_file):
                        state = load_state(state_file)
                        entry = state["issues"].get(str(w.issue_number), {})
                        now = datetime.now(UTC)
                        redispatch_at = _windowed_redispatch_at(
                            entry, window_minutes=config.watchdog.redispatch_window_minutes
                        ) + [now.isoformat().replace("+00:00", "Z")]
                        # Issue #783: a deterministic launch failure kind is a
                        # process failure, not a judgment call -- mechanical.
                        # Issue #807: a deterministic judgment failure kind
                        # (genuine local commits) is a judgment call -- judgment.
                        deterministic_judgment = (
                            failure_kind in DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS
                        )
                        reason_class = "judgment" if deterministic_judgment else "mechanical"
                        state = _escalate_issue(
                            state,
                            w.issue_number,
                            reason=failure_kind,
                            reason_class=reason_class,
                            issue_extra={"redispatch_at": redispatch_at},
                        )
                        state["issues"][str(w.issue_number)].pop("worker_pid", None)
                        state["issues"][str(w.issue_number)].pop("worker_process_start_time", None)
                        write_gate.save_state(state)
                        write_gate.transition(
                            gh,
                            config.labels,
                            w.issue_number,
                            _escalation_edge("redispatch_escalated", reason_class),
                        )
                        state = write_gate.append_event(
                            state,
                            "session_failed_escalated",
                            {
                                "issue_number": w.issue_number,
                                "failure_kind": failure_kind,
                                "removed_labels": sorted(active_labels),
                                "redispatch_count": len(redispatch_at),
                            },
                        )
                        write_gate.save_state(state)

            w.reap_sidecar(
                sessions_dir,
                api_config=config.api_worker,
                state_dir=state_file.parent,
            )
            reaped.append(
                {
                    "issue_number": w.issue_number,
                    "adapter_kind": w.adapter_kind,
                    "failure_kind": failure_kind,
                    "error": w.error,
                    "pid": w.pid,
                }
            )
            # Issue #295: a launch-failed rework session must still be returned to
            # rework_requested so its owning lane can re-dispatch it.
            _reap_restore_rework_requested(
                state_file,
                gh,
                config,
                open_prs_by_issue,
                w,
                failure_kind=failure_kind,
                repo_root=repo_root,
                write_gate=write_gate,
            )
            continue
        if not w.is_alive():
            # Issue #755: the confirmed-dead decision (including the
            # max_inconclusive_probe_deferrals grace period and counter) is now
            # owned by a single helper shared with reconcile.detect_drift.
            if not is_worker_confirmed_dead(
                w,
                config,
                now_for_health,
                sessions_dir,
                persist_inconclusive_probe_counter=persist_inconclusive_probe_counter,
            ):
                continue

            # Inspect the worktree before deciding how to classify and relabel.
            # This is the single enforcement point for issue #252.
            worktree_path = Path(w.worktree_path)
            inspection = inspect_worktree_state(
                worktree_path,
                config.dispatch.base_ref,
                config.dispatch.injected_paths,
                config.dispatch.materialize_dirs,
            )
            is_completed = inspection.state == WorktreeState.COMPLETED

            # Post-mortem extraction (issue #261) is intertwined with log-tail
            # classification. For a completed-but-unpublished worktree, we want
            # failure_kind to be "unpublished_work" even if the terminal log
            # tail would otherwise look like a tool-rejection (worker_blocked).
            # Call update_* first when completed, then run classify_and_record
            # for diagnostics (it will no-op on the sidecar because failure_kind
            # is already set). For non-completed sessions, preserve the original
            # ordering so worker_blocked still escalates.
            if is_completed:
                # session_completed=True (issue #656): the worktree inspection
                # just above is ground truth that this session produced real,
                # committable work -- it cannot also have died to a provider
                # quota/rate-limit/auth failure, so log-tail marker matching
                # (which would otherwise treat the session's own completion
                # summary prose as fair game) is skipped entirely.
                if w.adapter_kind == "devin":
                    failure_kind, throttled_until = (
                        update_session_record_with_failure_classification(
                            sessions_dir,
                            w.issue_number,
                            fallback_kind="unpublished_work",
                            config=config,
                            session_completed=True,
                            now=now_for_health,
                        )
                    )
                elif w.adapter_kind == "claude-code":
                    failure_kind, throttled_until = (
                        update_worker_record_with_failure_classification(
                            sessions_dir,
                            w.issue_number,
                            fallback_kind="unpublished_work",
                            config=config,
                            session_completed=True,
                            now=now_for_health,
                        )
                    )
                elif w.adapter_kind == "api":
                    failure_kind, throttled_until = (
                        update_worker_record_with_failure_classification(
                            sessions_dir,
                            w.issue_number,
                            fallback_kind="unpublished_work",
                            config=config,
                            adapter_kind="api",
                            session_completed=True,
                            now=now_for_health,
                        )
                    )
                else:
                    failure_kind, throttled_until = None, None
                # Diagnostic post-mortem; its worker_blocked verdict is ignored
                # because the worktree itself proves the work was completed.
                classify_and_record(sessions_dir, config, w, now=datetime.now(UTC))
            else:
                classify_and_record(sessions_dir, config, w, now=datetime.now(UTC))
                fallback_kind = "stalled" if inspection.state != WorktreeState.UNKNOWN else None
                if w.adapter_kind == "devin":
                    failure_kind, throttled_until = (
                        update_session_record_with_failure_classification(
                            sessions_dir,
                            w.issue_number,
                            fallback_kind=fallback_kind,
                            config=config,
                            now=now_for_health,
                        )
                    )
                elif w.adapter_kind == "claude-code":
                    failure_kind, throttled_until = (
                        update_worker_record_with_failure_classification(
                            sessions_dir,
                            w.issue_number,
                            fallback_kind=fallback_kind,
                            config=config,
                            now=now_for_health,
                        )
                    )
                elif w.adapter_kind == "api":
                    failure_kind, throttled_until = (
                        update_worker_record_with_failure_classification(
                            sessions_dir,
                            w.issue_number,
                            fallback_kind=fallback_kind,
                            config=config,
                            adapter_kind="api",
                            now=now_for_health,
                        )
                    )
                else:
                    failure_kind, throttled_until = None, None

            if failure_kind and throttled_until:
                # Update state with throttle window
                with state_lock(state_file):
                    state = load_state(state_file)
                    state = set_throttled_until(
                        state,
                        throttled_until,
                        reason=failure_kind,
                        adapter_kind=w.adapter_kind,
                    )
                    write_gate.save_state(state)

            # Reap the sidecar to prevent phantom sessions from PID recycling (issue #113)
            # Delete the sidecar file after the session is detected as dead and classified
            w.reap_sidecar(
                sessions_dir,
                api_config=config.api_worker,
                state_dir=state_file.parent,
            )
            reaped.append(
                {
                    "issue_number": w.issue_number,
                    "adapter_kind": w.adapter_kind,
                    "failure_kind": failure_kind,
                    "error": w.error,
                    "pid": w.pid,
                }
            )

            # Issue #1342: emit a distinct error-level event on the FIRST
            # detection of a provider account suspension so the operator learns
            # about a billing problem in minutes, not after the redispatch cap
            # drains. ``provider_suspended`` is terminal (no cooldown) and sits
            # in DETERMINISTIC_ESCALATION_FAILURE_KINDS, so the no-open-PR path
            # below escalates the issue to operator-queue on this same pass —
            # there is no redispatch, hence exactly one episode and no spam.
            # The sidecar was just reaped, so the next pass won't re-see this
            # worker — the event fires once per episode by construction.
            if w.adapter_kind == "api" and failure_kind == "provider_suspended":
                with state_lock(state_file):
                    state = load_state(state_file)
                    state = write_gate.append_event(
                        state,
                        "api_worker_provider_suspended",
                        {
                            "issue_number": w.issue_number,
                            "pid": w.pid,
                            "process_start_time": w.process_start_time,
                            "provider": w.provider,
                            "failure_kind": failure_kind,
                        },
                        level="error",
                    )
                    write_gate.save_state(state)

            # Issue #118: reconcile labels for dead sessions with no open PR
            if w.issue_number not in open_prs_by_issue:
                try:
                    issue = gh.issue_view(w.issue_number)
                except Exception:
                    # Issue may have been deleted or we lack access; skip relabel
                    continue
                issue_labels = label_names(issue)
                active_labels = issue_labels & config.labels.active
                # Gate the WHOLE reclaim on an active label actually being
                # present, matching reconcile.py's issue_active_label_no_open_pr
                # pattern (~536-580) so all three sites agree. An issue with
                # no active label -- e.g. one carrying only a terminal label
                # like agent:human-needed/agent:done/agent:blocked -- has
                # nothing here to reclaim; it must never get `ready` added
                # back just because it also has a stale
                # dispatched/dead-worker/no-PR state.json entry. (A prior
                # revision gated on "not active_labels and not needs_ready" to
                # also repair a remove-succeeded-but-add-failed partial
                # failure once the active label was already gone -- but that
                # made this lane indistinguishable from "issue is legitimately
                # terminal-only", which is the regression this gate now
                # avoids. `needs_ready` is still honored below whenever an
                # active label IS present, so the common
                # remove-and-add-together case is unaffected.)
                if not active_labels:
                    continue
                needs_ready = config.labels.ready not in issue_labels

                # Issue #252: completed-but-unpublished work takes the salvage
                # path (push + PR) instead of re-dispatching.
                # Issue #1130: relax the salvage trigger from ``is_completed``
                # (clean + ahead) to ``ahead_count > 0`` (ahead, regardless of
                # working-tree dirt). A worker that dies mid-push leaves a
                # committed-but-unpushed branch; the worktree may also carry
                # shim/scaffolding dirt (e.g. ``.devin/`` artifacts) that is
                # not in ``injected_paths`` and so reads as worker-authored
                # dirty, classifying the worktree PARTIAL instead of COMPLETED.
                # Salvage pushes the branch ref (the committed work), not the
                # working tree — the dirt is irrelevant to the push and survives
                # in the worktree for later inspection. Without this relaxation
                # the dead-session lane relabels to ready, the next redispatch
                # hits the unpushed commits, raises ``worktree_unsafe``, and
                # escalates to a human without ever attempting the cheap safe
                # action (salvage-the-commit).
                salvage_error: str | None = None
                has_salvageable_commits = inspection.ahead_count > 0
                if has_salvageable_commits and repo_root is not None:
                    salvaged, salvage_error = _attempt_salvage(
                        gh=gh,
                        config=config,
                        repo_root=repo_root,
                        worktree_path=worktree_path,
                        branch=w.branch,
                        base_ref=inspection.resolved_base_ref or "",
                        issue_number=w.issue_number,
                        active_labels=active_labels,
                        issue_labels=issue_labels,
                        state_file=state_file,
                        failure_kind=failure_kind,
                        issue_title=issue.get("title") if issue else None,
                        issue=issue,
                        write_gate=write_gate,
                    )
                    if salvaged:
                        continue
                    # Salvage failed: fall through to the normal relabel path below.

                # Issue #1244: cross-repo scope tripwire. Before relabeling
                # to ``automated-ready`` for another redispatch, check whether
                # the issue's title names another managed repo in the fleet.
                # A dead worker whose issue scope targets a sibling repo
                # hopped to that repo's worktree; redispatching repeats the
                # hop forever.  Override the failure_kind so the
                # DETERMINISTIC_ESCALATION_FAILURE_KINDS check below
                # escalates on the first occurrence instead of looping.
                scope_result = cross_repo_scope_gate(
                    str(issue.get("title") or ""),
                    str(issue.get("body") or ""),
                    sweep_dispatching_repo_name,
                    sweep_fleet_repos,
                )
                if not scope_result.passed:
                    failure_kind = "cross_repo_hop"

                # Track redispatch count for escalation cap (issue #165)
                # This relabel-to-ready path is a redispatch event
                with state_lock(state_file):
                    state = load_state(state_file)
                    entry = state["issues"].get(str(w.issue_number), {})
                    now = datetime.now(UTC)
                    redispatch_at = _windowed_redispatch_at(
                        entry, window_minutes=config.watchdog.redispatch_window_minutes
                    ) + [now.isoformat().replace("+00:00", "Z")]
                    # issue #261: a worker_blocked verdict (extracted from the
                    # Devin CLI's session store — see post_mortem.classify_and_record)
                    # means the worker was killed by a push-gate hook, not a
                    # generic stall/crash. Hot-redispatching it just repeats the
                    # same block, so it bypasses the redispatch-count cap entirely
                    # and escalates on the very first occurrence.
                    terminal_failure = failure_kind in DETERMINISTIC_ESCALATION_FAILURE_KINDS
                    # Issue #807: a deterministic judgment failure escalates
                    # immediately but as ``reason_class="judgment"``.
                    deterministic_judgment = (
                        failure_kind in DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS
                    )
                    immediate_escalation = terminal_failure or deterministic_judgment
                    if (
                        immediate_escalation
                        or len(redispatch_at) > config.watchdog.max_auto_redispatch
                    ):
                        # Escalate to human review instead of relabeling to ready
                        reason = (
                            failure_kind
                            if immediate_escalation and failure_kind is not None
                            else "redispatch_cap_exceeded"
                        )
                        # Issue #783: dead worker session / redispatch cap is a
                        # process failure, not a judgment call -- mechanical.
                        # Issue #807: a deterministic judgment failure (genuine
                        # local commits) is a judgment call -- judgment.
                        reason_class = "judgment" if deterministic_judgment else "mechanical"
                        state = _escalate_issue(
                            state,
                            w.issue_number,
                            reason=reason,
                            reason_class=reason_class,
                            issue_extra={"redispatch_at": redispatch_at},
                        )
                        # Issue #282: preserve the liveness fingerprint for the
                        # recovery path. The PID is already verified dead by the
                        # time we reach this branch, but clearing it removes the
                        # only signal the recovery probe can cross-check.
                        write_gate.save_state(state)
                        write_gate.transition(
                            gh,
                            config.labels,
                            w.issue_number,
                            _escalation_edge("redispatch_escalated", reason_class),
                        )
                        state = write_gate.append_event(
                            state,
                            "session_failed_escalated",
                            {
                                "issue_number": w.issue_number,
                                "failure_kind": failure_kind,
                                "removed_labels": sorted(active_labels),
                                "redispatch_count": len(redispatch_at),
                            },
                        )
                        write_gate.save_state(state)
                        continue
                    else:
                        entry["redispatch_at"] = redispatch_at
                        state["issues"][str(w.issue_number)] = entry
                        write_gate.save_state(state)
                # Remove all active labels and ensure ready label is present.
                # Issue #417: check (and record) the bool return values instead
                # of silently discarding them. A False here means this pass's
                # label swap did not fully land -- the issue remains eligible
                # for _detect_and_handle_orphaned_workers' no-open-PR sweep to
                # finish the reclaim on a later pass, since that lane
                # re-derives "does this still need fixing" from GitHub's live
                # label state every pass rather than from any flag written
                # here (and never touches redispatch_at, so a retry there
                # cannot double-count this as a second redispatch event).
                label_write_ok = True
                for label in sorted(active_labels):
                    if not gh.remove_issue_label(w.issue_number, label):
                        label_write_ok = False
                if needs_ready:
                    if not gh.add_issue_label(w.issue_number, config.labels.ready):
                        label_write_ok = False
                # Record the relabel event
                with state_lock(state_file):
                    state = load_state(state_file)
                    # Issue #282: preserve the liveness fingerprint so the
                    # recovery path can verify the worker is dead before removing
                    # the worktree, even after the session is classified as dead.
                    state = _emit_session_failed_relabeled(
                        state,
                        issue_number=w.issue_number,
                        reason="dead_worker_no_open_pr",
                        failure_kind=failure_kind,
                        removed_labels=sorted(active_labels),
                        added_ready=needs_ready,
                        label_write_ok=label_write_ok,
                        salvage_failed=has_salvageable_commits,
                        salvage_error=salvage_error,
                        state_path=state_file,
                        write_gate=write_gate,
                    )
                    write_gate.save_state(state)
            else:
                # Issue #295: open PR with request_changes or rework prompt =>
                # restore to rework_requested for its owning lane.
                #
                # Issue #315 finding 1: a completed worktree (ahead of base and
                # clean) proves the worker finished its work — even if this
                # reap pass's PR-list snapshot (fetched once, at line ~889)
                # hasn't caught up to a fresh push yet. Never roll a worker
                # that finished (and, per the open-PR guard above, already has
                # a PR reflecting or about to reflect that work) back to
                # rework_requested just because a launch-failure classifier
                # ran on its now-dead sidecar.
                if not is_completed:
                    pr_data = _rework_pr_for_worker(open_prs_by_issue, w)
                    if pr_data is not None:
                        pr_number = int(pr_data["number"])
                        try:
                            pr_view = gh.pr_view(pr_number)
                        except Exception:
                            pr_view = None
                        enriched = pr_view if pr_view else pr_data
                        is_candidate, reason = _is_pre_review_rework_candidate(
                            enriched, config, now_for_health
                        )
                        if is_candidate:
                            _route_dead_worker_to_pre_review_rework(
                                state_file,
                                gh,
                                config,
                                enriched,
                                w.issue_number,
                                reason,
                                failure_kind=failure_kind,
                                write_gate=write_gate,
                            )
                        else:
                            _reap_restore_rework_requested(
                                state_file,
                                gh,
                                config,
                                open_prs_by_issue,
                                w,
                                failure_kind=failure_kind,
                                repo_root=repo_root,
                                write_gate=write_gate,
                            )
                    else:
                        _reap_restore_rework_requested(
                            state_file,
                            gh,
                            config,
                            open_prs_by_issue,
                            w,
                            failure_kind=failure_kind,
                            repo_root=repo_root,
                            write_gate=write_gate,
                        )

    return reaped


def _safe_repo_slug(gh: GitHubLike) -> str:
    """Return the ``owner/repo`` slug, or ``"?"`` if the lookup fails.

    ``name_with_owner()`` raises ``GitHubError`` on failure (offline, gh
    missing, etc.); this is used only to qualify a closing-reference line, so
    a lookup failure must not stop salvage-PR creation. Mirrors
    ``reconcile._repo_slug``.
    """
    try:
        return gh.name_with_owner()
    except Exception:
        return "?"


def _dispatching_repo_name(gh: GitHubLike, repo_root: Path) -> str:
    """Return the repo-name segment of the dispatching repo (issue #1244).

    Prefers ``gh.name_with_owner()`` (``owner/repo``) so the name matches
    the fleet registry's keys.  Falls back to ``repo_root.name`` (the
    directory name) when the GitHub lookup fails (offline, gh missing) —
    the directory name is usually the same as the GitHub repo name, and a
    mismatch only means the scope gate cannot attribute the issue, which
    is the safe direction (pass, not block).
    """
    try:
        nwo = gh.name_with_owner()
        parts = nwo.rsplit("/", 1)
        return parts[1] if len(parts) == 2 else nwo
    except Exception:
        return repo_root.name


def _open_salvage_pr(
    *,
    gh: GitHubLike,
    config: OrchestratorConfig,
    repo_root: Path | None,
    branch: str,
    base_ref: str,
    issue_number: int,
    active_labels: set[str],
    issue_labels: set[str],
    issue_title: str | None = None,
    source_description: str = "worker branch",
    state_file: Path | None = None,
) -> tuple[int | None, str | None, ValidationResult | None]:
    """Open a PR for a salvaged worker branch and move issue labels toward ``pr_open``.

    Returns ``(pr_number, error, closing_ref)``. ``pr_number`` is the created
    PR number, or ``None`` when the PR could not be created. ``error`` is
    ``None`` when both the PR and the label swap succeeded; otherwise it
    describes the first failure encountered (a missing ``repo_root``, a
    failed PR create, or a label write failure after the PR was created).
    ``closing_ref`` is the `~charlie_work.closing_reference.ValidationResult`
    from canonicalizing the closing-reference line before the PR was
    created, or ``None`` when PR creation never reached that far (missing
    ``repo_root``).

    cw#1263: the body's ``Closes #N`` line is validated/canonicalized via
    `closing_reference.validate_closing_reference` before ``gh.pr_create``
    ever sees it -- this is the sole point where both salvage/orphan-recovery
    callers (`_attempt_salvage`, `_open_pr_for_orphaned_branch`) create a PR,
    so routing the fixed-up body through here covers both without a second
    call site to keep in sync. After a successful create, GitHub's own
    ``closingIssuesReferences`` resolution is queried and compared against
    ``issue_number``; a mismatch is logged (``pr_closing_ref_unlinked``) but
    never blocks the return -- this is the only verification surface that
    would catch it, since GitHub's own auto-close resolution can diverge from
    the text charlie-work wrote even when that text looks correct.
    """
    if repo_root is None:
        return None, "repo_root is required to open a salvage PR", None

    base_branch = resolve_base_branch_name(repo_root, base_ref)

    title = (
        f"Salvaged work for #{issue_number}: {issue_title}"
        if issue_title
        else f"Salvaged work for issue #{issue_number}"
    )
    # The body must satisfy the same janitor gate as a worker-authored one
    # (`review.require_tests_or_rationale`). A fixed boilerplate string cannot:
    # it carries no rationale token, so every salvage PR failed a gate on text
    # the orchestrator itself wrote. Derive the rationale from the worker's own
    # commit log instead of injecting the gate's keywords -- a branch with no
    # commits still yields no summary, and still correctly fails.
    body = f"Closes #{issue_number}\n\nSalvaged by the orchestrator from a {source_description}."
    # Pass the RESOLVED base branch, not the raw ``base_ref``. The orphaned-branch
    # lane (``_open_pr_for_orphaned_branch``) sources ``base_ref`` straight from
    # ``config.dispatch.base_ref``, whose default is ``""`` and which the live
    # config leaves unset -- so production reaches here with the empty sentinel.
    # ``require_valid_rev("")`` raises, ``summarize_branch_work`` returns "", and
    # the body falls back to boilerplate that cannot pass the janitor gate: the
    # exact defect this code exists to fix, on the lane that hits it most.
    summary = summarize_branch_work(
        repo_root,
        branch,
        base_branch,
        test_path_globs=config.test_adequacy.test_path_globs,
    )
    if summary:
        body = f"{body}\n\n{summary}"

    closing_ref = validate_closing_reference(body, issue_number, repo=_safe_repo_slug(gh), gh=gh)
    body = closing_ref.body
    if closing_ref.changed and state_file is not None:
        log_event(
            state_file,
            "pr_closing_ref_rewritten",
            {
                "issue_number": issue_number,
                "findings": list(closing_ref.findings),
                "source": source_description,
            },
        )

    # cw#1273: every gh.pr_create call site routes through the bounded outer
    # retry + duplicate-PR guard instead of calling gh.pr_create directly.
    retry_result = create_pr_with_retry(
        gh,
        head=branch,
        base=base_branch,
        title=title,
        body=body,
        max_retries=config.runtime.pr_create_retry_max_attempts,
        base_seconds=config.runtime.pr_create_retry_base_seconds,
    )
    pr_number = retry_result.pr_number
    if pr_number is None:
        return (
            None,
            retry_result.error or "gh pr create failed or returned no PR number",
            closing_ref,
        )

    # `pr_number` is falsy (0) under `dry_run`, where no real PR was opened and
    # a `gh pr view 0` call would be both wasted and nonsensical -- only probe
    # a real, truthy PR number.
    if pr_number and state_file is not None:
        query_ok = True
        try:
            pr_view = gh.pr_view(pr_number, fields=PR_CLOSING_ISSUES_FIELDS)
        except Exception:
            pr_view = {}
            query_ok = False
        linked_numbers = closing_issues_referenced_numbers(pr_view)
        # Only log when the query itself succeeded: a transient `gh` failure
        # collapses to the same empty result as "GitHub really didn't link
        # the issue", and this event exists to be acted on -- conflating a
        # failed probe with a genuine miss would make it noisy and untrustworthy.
        if query_ok and issue_number not in linked_numbers:
            log_event(
                state_file,
                "pr_closing_ref_unlinked",
                {
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "linked_issue_numbers": sorted(linked_numbers),
                },
            )

    label_write_ok = True
    for label in sorted(active_labels):
        if not gh.remove_issue_label(issue_number, label):
            label_write_ok = False
    if config.labels.pr_open not in issue_labels:
        if not gh.add_issue_label(issue_number, config.labels.pr_open):
            label_write_ok = False

    if not label_write_ok:
        return pr_number, "PR created but label write failed", closing_ref

    return pr_number, None, closing_ref


def _salvage_already_landed(
    *,
    gh: GitHubLike,
    config: OrchestratorConfig,
    repo_root: Path,
    branch: str,
    base_ref: str,
    issue_number: int,
    issue: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    """Return ``(already_landed, reason)`` if salvage should be skipped.

    Issue #1221 / #1241: a dead session's snapshot (issue, pr_number, branch)
    can be stale by the time staleness trips and salvage fires -- the linked
    issue may have been closed and/or its PR merged by an operator or sibling
    worker inside the staleness threshold window, and (the #1241 race) the
    branch's commits may already be reachable from origin/main via a merge
    commit whose tree differs from the salvage head's tree. Re-check LIVE
    terminal state at fire time instead of trusting the snapshot.

    This is now a thin delegate to the shared single enforcement point
    ``salvage_superseded.check_salvage_superseded`` so the workflow salvage
    lane and the reconcile salvage lane cannot diverge on which checks fire
    or in which order. The shared check covers:

    1. the linked issue is CLOSED (``issue`` carries ``state`` from the
       caller's ``gh.issue_view`` -- one call, already made; if ``issue`` is
       None the shared check fetches it via ``gh.issue_view`` so the
       closed-issue check still fires on the reconcile lane, which had not
       fetched).
    2. a PR binding to this issue is MERGED (``gh.merged_prs_for_issue`` -- one
       call). A failed search (``ok=False``) is treated as "unknown", which
       falls through to opening the PR; a human reviews salvage PRs anyway.
    3. the salvage branch's tree contributes an empty diff against current main
       (``salvage_branch_empty_diff`` -- a fetch + two rev-parse calls). This
       is the belt-and-suspenders for the case where (1)/(2) miss (e.g. a
       squash-merge that closed the issue but whose PR search lags, or work
       landed via a sibling branch). Fails safe (returns False) on git error.
    4. (#1241) the salvage branch's tip is an ANCESTOR of origin/main
       (``salvage_branch_reachable_from_main`` -- a fetch + ``git merge-base
       --is-ancestor``). This catches the #1241 race that (3) misses: a merge
       commit incorporated the salvage head while main advanced with other
       commits, so the trees differ (empty-diff reads "not empty") but the
       salvage head carries nothing new (ancestry reads "already on main").
       Fails open on git error.

    ``reason`` is a short string identifying which check fired, recorded in the
    skip event (``salvage_skipped_already_landed`` for reasons 1-3,
    ``salvage_skipped_superseded`` for reason 4) for diagnosis.
    """
    return check_salvage_superseded(
        gh=gh,
        config=config,
        repo_root=repo_root,
        branch=branch,
        base_ref=base_ref,
        issue_number=issue_number,
        issue=issue,
    )


def _attempt_salvage(
    *,
    gh: GitHubLike,
    config: OrchestratorConfig,
    repo_root: Path,
    worktree_path: Path,
    branch: str,
    base_ref: str,
    issue_number: int,
    active_labels: set[str],
    issue_labels: set[str],
    state_file: Path,
    failure_kind: str | None,
    issue_title: str | None = None,
    issue: dict[str, Any] | None = None,
    write_gate: WriteGate,
) -> tuple[bool, str | None]:
    """Push a completed branch and open a PR, then move labels to ``pr_open``.

    Returns ``(ok, error)``. Errors are recorded as values and never raised.
    ``ok`` is ``True`` once the PR is created, even if the label swap failed;
    in that case ``error`` describes the label failure and the
    ``session_salvaged`` event records ``label_write_ok=False``.

    ``ok`` is also ``True`` (with ``error=None``) when salvage is *skipped*
    because the work already landed (issue #1221): the dead session's snapshot
    can be stale, so before opening a PR we re-check live terminal state. A
    skip emits ``salvage_skipped_already_landed`` instead of opening a vestigial
    duplicate PR, and the caller treats it as "handled" (no redispatch).

    NOTE (issue #1326): the ``push_branch`` call below threads
    ``dry_run=write_gate.dry_run`` so a dry-run invocation does not issue a
    real ``git push``. This is explicit-threading (mirroring
    ``_reconcile_locked``'s convention) rather than a 7th WriteGate primitive
    -- gating an external git push through WriteGate was an open design
    question (see issue #1326's remedy section) and the simplest correct
    fix is to thread the flag directly. Downstream state writes and label
    transitions remain gated by ``write_gate``; the ``gh pr create`` in
    ``_open_salvage_pr`` is gated at the ``GitHub`` client sink level.
    """
    write_gate = require_write_gate(write_gate)
    already_landed, skip_reason = _salvage_already_landed(
        gh=gh,
        config=config,
        repo_root=repo_root,
        branch=branch,
        base_ref=base_ref,
        issue_number=issue_number,
        issue=issue,
    )
    if already_landed:
        with state_lock(state_file):
            state = load_state(state_file)
            # Issue #282: preserve the liveness fingerprint so the recovery
            # path can verify the worker is dead before the worktree is
            # reclaimed.
            # Issue #1241: the event kind is mapped from the skip reason so the
            # new reachability skip (``commits_reachable``) emits
            # ``salvage_skipped_superseded`` while the #1221 reasons keep their
            # existing ``salvage_skipped_already_landed`` event.
            state = write_gate.append_event(
                state,
                salvage_skip_event_kind(
                    skip_reason
                ),  # event-consumer: audit-only -- kind resolves to one of two registered literals (salvage_skipped_already_landed / salvage_skipped_superseded), both in _LEVEL_BY_KIND; the actionable state mutation (worktree reap) happens in the caller, this event is the observable skip record (issue #1241)
                {
                    "issue_number": issue_number,
                    "failure_kind": failure_kind,
                    "reason": skip_reason,
                    # The skip path does NOT remove labels -- label cleanup is the
                    # reconcile lane's job. Record the active labels at skip time
                    # for diagnosis (what state the issue was in), not as removed.
                    "active_labels": sorted(active_labels),
                },
            )
            write_gate.save_state(state)
        return True, None

    push_ok, push_error = push_branch(
        repo_root, branch, worktree_path=worktree_path, dry_run=write_gate.dry_run
    )
    if not push_ok:
        return False, push_error

    pr_number, pr_error, _closing_ref = _open_salvage_pr(
        gh=gh,
        config=config,
        repo_root=repo_root,
        branch=branch,
        base_ref=base_ref,
        issue_number=issue_number,
        active_labels=active_labels,
        issue_labels=issue_labels,
        issue_title=issue_title,
        source_description="completed-but-unpublished worker worktree",
        state_file=state_file,
    )
    if pr_number is None:
        return False, pr_error or "gh pr create failed or returned no PR number"

    with state_lock(state_file):
        state = load_state(state_file)
        # Issue #282: preserve the liveness fingerprint so the recovery path
        # can verify the worker is dead before the worktree is reclaimed.
        state = write_gate.append_event(
            state,
            "session_salvaged",
            {
                "issue_number": issue_number,
                "failure_kind": failure_kind,
                "removed_labels": sorted(active_labels),
                "pr_number": pr_number,
                "label_write_ok": pr_error is None,
                "label_error": pr_error,
            },
        )
        write_gate.save_state(state)
    return True, pr_error


def _open_pr_for_orphaned_branch(
    *,
    gh: GitHubLike,
    config: OrchestratorConfig,
    repo_root: Path | None,
    branch: str,
    base_ref: str,
    issue_number: int,
    active_labels: set[str],
    issue_labels: set[str],
    issue_title: str | None = None,
    state_file: Path | None = None,
) -> tuple[int | None, str | None, ValidationResult | None]:
    """Open a PR for a branch that the worker pushed but could not create a PR for.

    Returns ``(pr_number, error, closing_ref)``. Errors are recorded as
    values and never raised. This is the orchestrator-side recovery for
    issue #935: workers are unauthenticated in their environment, so after
    pushing a completed branch they cannot run ``gh pr create``. The
    orchestrator, which is authenticated, creates the PR and moves the issue
    labels toward ``pr_open``. See `_open_salvage_pr` for the closing-
    reference validation and post-create verification this delegates to.
    """
    return _open_salvage_pr(
        gh=gh,
        config=config,
        repo_root=repo_root,
        branch=branch,
        base_ref=base_ref,
        issue_number=issue_number,
        active_labels=active_labels,
        issue_labels=issue_labels,
        issue_title=issue_title,
        source_description="worker branch that could not open a PR",
        state_file=state_file,
    )


def _issues_with_live_workers(sessions_dir: Path) -> set[int]:
    """Return the set of issue numbers that have currently alive worker sessions.

    Reads session sidecar files from both devin-shell and claude-code adapters,
    then checks each record's PID liveness using the adapter-specific liveness
    probe. Returns the set of issue numbers with alive PIDs.
    """
    from .worker import iter_workers

    return {w.issue_number for w in iter_workers(sessions_dir) if w.is_alive()}
