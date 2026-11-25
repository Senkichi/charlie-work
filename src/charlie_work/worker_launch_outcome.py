"""What both worker lanes record once a launch has returned (issues #3098, #3099, #3100).

Fresh ``dispatch`` and remote ``dispatch_rework`` each used to carry their own
copy of the bookkeeping that follows a launch. This module owns the shared part,
for both lanes; it lives at the package root, beside
:mod:`charlie_work.worker_launch_gate`, because every top-level function defined
under ``orchestration/`` is auto-installed on ``OrchestratorApp``.

Interface: launch results and lane inputs in, recorded outcome out.

- :func:`launch_base_entry` builds the issue record a launch starts from (the
  stored record plus number, title, url, branch and prompt, minus the claim stamp
  and any ``label_error``).
- :func:`record_successful_launch` completes that record for a launch that
  succeeded: ``status`` and ``dispatched_at`` as the lane decided them, the
  orphan flags and the previous death's failure kind cleared, the launched
  worker's pid stamped from the launch ``result``.
- :func:`record_failed_launch` is the one entry point for a failed launch. It
  picks the blocked-environment or the generic branch itself, emits the branch's
  Events through the :class:`WriteGate` it is given, and returns a
  :class:`LaunchFailureOutcome`.
- :func:`record_label_error` records a label transition that did not apply.
- The reap helpers decide, stamp the issue entry they were handed and return an
  Event payload (or ``None`` when no reap happened).

What the lanes keep: every label transition (``_wf.transition``, and the
escalation edge a failure returns), seating the entry into ``state`` and the
saves (fresh persists once after its own tail; rework persists inline), and the
lane-only steps (fresh's escalation clear, failure-counter pop and manual
harness status; rework's PR failure-marker reset, redispatch counter and cap).

Shape. The module imports nothing from :mod:`charlie_work.workflow`. Names that
tests replace on ``charlie_work.workflow`` by path (``_try_reap_blocked_foreign_writer``
and friends, ``utc_now``) stay resolved through ``workflow`` in the lane: the lane
hands in the value (``dispatched_at``) or a zero-argument callable (the reap), so
those patches keep taking effect. Lane differences are explicit keyword arguments,
never inferred, and the module never seats an entry into ``state`` or mutates the
stored record.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from charlie_work.adapters import SessionDispatchResult
from charlie_work.config import (
    DETERMINISTIC_ESCALATION_FAILURE_KINDS,
    DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS,
    PRE_LAUNCH_BLOCKED_ENVIRONMENT_FAILURE_KINDS,
)
from charlie_work.dispatch_selection import (
    _windowed_blocked_environment_at,
    _windowed_foreign_writer_reaps,
    _windowed_redispatch_at,
)
from charlie_work.escalation import _escalate_issue
from charlie_work.labels import TransitionOutcome, TransitionResult
from charlie_work.state import (
    clear_dead_worker_failure_kind,
    clear_escalation,
    clear_escalation_on_issue_prs,
)
from charlie_work.worker_pid_stamp import stamp_worker_process
from charlie_work.write_gate import WriteGate, require_write_gate


def launch_base_entry(
    stored: dict[str, Any],
    *,
    issue_number: int,
    title: Any,
    url: Any,
    branch_name: str,
    prompt_path: Any,
) -> dict[str, Any]:
    """The issue record a launch outcome starts from, as a new dict.

    ``stored`` spread first, then this launch's number, title, url, branch and
    prompt; the claim stamp (``dispatch_pending_at``) and any ``label_error`` are
    dropped. ``stored`` itself is not touched and the result is not seated in
    ``state``.
    """
    entry = {
        **stored,
        "number": issue_number,
        "title": title,
        "url": url,
        "branch_name": branch_name,
        "prompt_path": str(prompt_path),
    }
    entry.pop("dispatch_pending_at", None)
    entry.pop("label_error", None)
    return entry


def record_successful_launch(
    entry: dict[str, Any],
    *,
    status: str,
    dispatched_at: str,
    result: SessionDispatchResult | None,
) -> dict[str, Any]:
    """Complete ``entry`` (from :func:`launch_base_entry`) for a launch that succeeded.

    A success supersedes any previous orphan flag and the previous death's
    failure kind (issue #1917), records ``status`` and ``dispatched_at`` (the lane
    reads its clock and decides the status), and stamps the worker pid and process
    start time from the launch ``result`` (issue #207), dropping a previous
    epoch's pair when the launch reports none. Mutates and returns ``entry``.
    """
    entry.pop("orphan_flagged_at", None)
    entry.pop("orphan_drift_fingerprint", None)
    entry.pop("orphan_drift_at", None)
    clear_dead_worker_failure_kind(entry)
    entry["status"] = status
    entry["dispatched_at"] = dispatched_at
    stamp_worker_process(entry, result)
    return entry


def label_error_reason(label_error: dict[str, Any]) -> str:
    """Format a persisted label_error dict into a short, human-readable reason."""
    edge = label_error.get("edge", "unknown")
    outcome = label_error.get("outcome", "unknown")
    add_failures = label_error.get("add_failures") or []
    remove_failures = label_error.get("remove_failures") or []
    add_labels = [label for _issue, label in add_failures]
    remove_labels = [label for _issue, label in remove_failures]
    parts = [f"label transition '{edge}' {outcome}"]
    if add_labels:
        parts.append(f"add failures: {add_labels}")
    if remove_labels:
        parts.append(f"remove failures: {remove_labels}")
    return "; ".join(parts)


def record_label_error(
    write_gate: WriteGate,
    state: dict[str, Any],
    entry: dict[str, Any],
    result: TransitionResult,
    issue_number: int,
    edge: str,
    *,
    label_errors: list[int],
    label_error_failures: dict[int, str],
) -> None:
    """Record a label transition that did not apply, then persist ``state``.

    A transition that returns ``TransitionOutcome.APPLIED`` records nothing and
    writes nothing. Otherwise the ``label_error`` dict (edge, outcome, add and
    remove failures) is stored on ``entry``, ``issue_number`` joins
    ``label_errors``, its reason joins ``label_error_failures``, and the state
    is saved through ``write_gate``. ``entry`` must already be the record held
    in ``state["issues"]`` (or the lane's own copy of it, as it was before this
    helper existed): this function never re-seats it.
    """
    write_gate = require_write_gate(write_gate)
    if result.outcome == TransitionOutcome.APPLIED:
        return
    label_error = {
        "edge": edge,
        "outcome": result.outcome.value,
        "add_failures": result.add_failures,
        "remove_failures": result.remove_failures,
    }
    entry["label_error"] = label_error
    label_errors.append(issue_number)
    label_error_failures[issue_number] = label_error_reason(label_error)
    write_gate.save_state(state)


def _stamp_reap(entry: dict[str, Any], prior_reaps: list[str], now_iso: str) -> None:
    """Reset the blocked-environment counter and record the reap in the window."""
    entry["blocked_environment_at"] = []
    entry["foreign_writer_reaps"] = prior_reaps + [now_iso]


def reap_blocked_foreign_writer_at_cap(
    entry: dict[str, Any],
    *,
    prior_reaps: list[str],
    max_reaps: int,
    try_reap: Callable[[], bool],
    now_iso: str,
    issue_number: int,
    failure_kind: str | None,
    pid: int | None,
    blocked_environment_count: int,
) -> dict[str, Any] | None:
    """Reap a blocked foreign writer instead of escalating (issue #1423).

    Called on a launch failure whose blocked-environment counter has passed the
    redispatch cap. ``prior_reaps`` is the windowed reap history: once it holds
    ``max_reaps`` entries the issue escalates rather than reaping again (a
    persistently-blocked worktree must not loop between reap and redispatch).
    ``try_reap`` is only called under the cap. On a successful reap ``entry`` is
    stamped (``blocked_environment_at`` cleared, ``foreign_writer_reaps``
    extended with ``now_iso``) and the Event payload is returned for the caller to
    emit; otherwise ``entry`` is untouched and ``None`` is returned.
    """
    if not (len(prior_reaps) < max_reaps and try_reap()):
        return None
    _stamp_reap(entry, prior_reaps, now_iso)
    return {
        "issue_number": issue_number,
        "failure_kind": failure_kind,
        "pid": pid,
        "blocked_environment_count": blocked_environment_count,
        "foreign_writer_reap_count": len(prior_reaps) + 1,
    }


def stamp_pre_claim_reap(
    entry: dict[str, Any],
    *,
    prior_reaps: list[str],
    now_iso: str,
    issue_number: int,
    pid: int | None,
    blocked_environment_count: int,
) -> dict[str, Any]:
    """Record a reap made by rework's pre-claim safety net (issue #1423).

    The reap itself has already happened (the lane decided it, against its own
    patched ``_reap_idle_foreign_writer``); this stamps ``entry`` as
    :func:`reap_blocked_foreign_writer_at_cap` does and returns the Event
    payload, which carries no ``failure_kind`` because no launch failed.
    """
    _stamp_reap(entry, prior_reaps, now_iso)
    return {
        "issue_number": issue_number,
        "pid": pid,
        "blocked_environment_count": blocked_environment_count,
        "foreign_writer_reap_count": len(prior_reaps) + 1,
    }


@dataclass(frozen=True)
class LaunchFailureOutcome:
    """What :func:`record_failed_launch` decided for one failed launch.

    ``state`` is the state dict after the call (the Events it emitted, the
    escalation it applied). ``entry`` is the issue record: on an escalation the
    record stored in ``state["issues"]`` itself (so ``record_label_error`` can
    persist onto it), otherwise the lane-built entry that was passed in.
    ``status`` is what the lane records (``"escalated"`` or the restored status);
    a failed launch always records ``dispatched_at = None``. ``reason_class`` is
    set exactly when the launch escalated, and the lane then applies the
    ``redispatch_escalated`` edge for it.
    """

    state: dict[str, Any]
    entry: dict[str, Any]
    status: str
    reason_class: str | None


def _seat(
    state: dict[str, Any], entry: dict[str, Any], issue_number: int, *, restored_status: str
) -> None:
    """Rework's inline restore: stamp the status, seat the entry in ``state``."""
    entry["status"] = restored_status
    entry["dispatched_at"] = None
    state["issues"][str(issue_number)] = entry


def record_failed_launch(
    write_gate: WriteGate,
    state: dict[str, Any],
    entry: dict[str, Any],
    *,
    failed_result: SessionDispatchResult | None,
    issue_number: int,
    now: datetime,
    window_minutes: int,
    max_auto_redispatch: int,
    max_reaps: int,
    try_reap: Callable[[], bool],
    recent_attempts: Callable[[list[str]], list[str]] | None,
    rework: bool,
    save_inline: bool,
    clear_stale_escalation: bool,
    escalation_carries_entry: bool,
    z_stamps: bool,
    restored_status: str,
    previous_status: str,
    cap_reason: str,
) -> LaunchFailureOutcome:
    """Record one failed launch, in either lane (issues #1393, #1423, #461, #515, #807).

    A pre-launch environment block (``PRE_LAUNCH_BLOCKED_ENVIRONMENT_FAILURE_KINDS``)
    never started a worker, so it counts against ``blocked_environment_at``, not
    the failure counter: at ``max_auto_redispatch`` it first tries one bounded
    foreign-writer reap (``try_reap``, only while the windowed reap history is
    under ``max_reaps``) and otherwise escalates ``dispatch_blocked_environment``.
    Any other failure counts against the lane's failure counter and escalates
    immediately for a deterministic kind, or past the cap with ``cap_reason``.

    The lane differences are declared inputs:

    - ``rework``: the lane (Event kind prefix, and the failure counter -- rework's
      windowed ``redispatch_at`` versus fresh's never-pruned ``dispatch_failed_at``,
      which ``recent_attempts`` windows for the cap decision only).
    - ``restored_status`` / ``previous_status`` / ``cap_reason``: the status after a
      restore, the ``previous_status`` of the ``session_failed_escalated`` Event and
      the escalation reason of the cap.
    - ``save_inline``: persist here (and seat ``entry`` in ``state``) versus leave
      both to the lane.
    - ``clear_stale_escalation``: clear a standing escalation on every restore.
    - ``escalation_carries_entry``: escalate with the whole lane-built entry as
      ``issue_extra`` versus only the counter and ``dispatched_at``.
    - ``z_stamps``: the failure counter's stamp format (``Z`` versus ``+00:00``).

    ``entry`` is never seated into ``state`` before an escalation: the escalation
    merges onto the stored record, which keeps the claim stamp and any stale
    ``label_error`` -- on both sides of the move.
    """
    write_gate = require_write_gate(write_gate)
    failure_kind = failed_result.failure_kind if failed_result is not None else None
    if failure_kind in PRE_LAUNCH_BLOCKED_ENVIRONMENT_FAILURE_KINDS:
        return _record_blocked_environment(
            write_gate,
            state,
            entry,
            failed_result=failed_result,
            failure_kind=failure_kind,
            issue_number=issue_number,
            now=now,
            window_minutes=window_minutes,
            max_auto_redispatch=max_auto_redispatch,
            max_reaps=max_reaps,
            try_reap=try_reap,
            rework=rework,
            save_inline=save_inline,
            clear_stale_escalation=clear_stale_escalation,
            escalation_carries_entry=escalation_carries_entry,
            restored_status=restored_status,
            previous_status=previous_status,
        )
    return _record_generic_failure(
        write_gate,
        state,
        entry,
        failure_kind=failure_kind,
        issue_number=issue_number,
        now=now,
        window_minutes=window_minutes,
        max_auto_redispatch=max_auto_redispatch,
        recent_attempts=recent_attempts,
        rework=rework,
        save_inline=save_inline,
        clear_stale_escalation=clear_stale_escalation,
        escalation_carries_entry=escalation_carries_entry,
        z_stamps=z_stamps,
        restored_status=restored_status,
        cap_reason=cap_reason,
    )


def _record_blocked_environment(
    write_gate: WriteGate,
    state: dict[str, Any],
    entry: dict[str, Any],
    *,
    failed_result: SessionDispatchResult | None,
    failure_kind: str | None,
    issue_number: int,
    now: datetime,
    window_minutes: int,
    max_auto_redispatch: int,
    max_reaps: int,
    try_reap: Callable[[], bool],
    rework: bool,
    save_inline: bool,
    clear_stale_escalation: bool,
    escalation_carries_entry: bool,
    restored_status: str,
    previous_status: str,
) -> LaunchFailureOutcome:
    now_iso = now.isoformat().replace("+00:00", "Z")
    blocked_environment_at = _windowed_blocked_environment_at(
        entry, window_minutes=window_minutes
    ) + [now_iso]
    blocking_error = failed_result.error if failed_result else None
    entry["blocked_environment_at"] = blocked_environment_at
    if len(blocked_environment_at) > max_auto_redispatch:
        # Issue #1423: before escalating a blocked-environment cap exhaustion for
        # a foreign writer, attempt to reap it one more time. A writer that was
        # active on earlier passes but has since gone idle is reaped here instead
        # of escalating a zombie to a human. Escalation is reserved for a writer
        # that is alive *and* active. The number of auto-reaps per issue is
        # bounded: each reap resets ``blocked_environment_at`` to ``[]``, so
        # without the windowed ``foreign_writer_reaps`` counter (which survives
        # the reset) a persistently-blocked worktree would loop forever between
        # reap and redispatch.
        reap_event = reap_blocked_foreign_writer_at_cap(
            entry,
            prior_reaps=_windowed_foreign_writer_reaps(entry, window_minutes=window_minutes),
            max_reaps=max_reaps,
            try_reap=try_reap,
            now_iso=now_iso,
            issue_number=issue_number,
            failure_kind=failure_kind,
            pid=failed_result.pid if failed_result else None,
            blocked_environment_count=len(blocked_environment_at),
        )
        if reap_event is not None:
            if clear_stale_escalation:
                clear_escalation(entry)
                clear_escalation_on_issue_prs(state, issue_number)
            if save_inline:
                _seat(state, entry, issue_number, restored_status=restored_status)
            state = write_gate.record_event(  # event-consumer: audit-only -- records a foreign-writer reap (issue #1423) already enforced by the blocked_environment_at reset and foreign_writer_reaps counter; consumed by tests/test_charlie_work.py regression tests.
                state,
                "rework_dispatch_blocked_environment_reaped"
                if rework
                else "dispatch_blocked_environment_reaped",
                reap_event,
            )
            if save_inline:
                write_gate.save_state(state)
            return LaunchFailureOutcome(state, entry, restored_status, None)
        # Escalate with the environment reason and the blocking path so the
        # operator sees "remove C:\...\wt", not "worker quality cap exceeded."
        state = _escalate_issue(
            state,
            issue_number,
            reason="dispatch_blocked_environment",
            reason_class="mechanical",
            issue_extra=entry
            if escalation_carries_entry
            else {"blocked_environment_at": blocked_environment_at, "dispatched_at": None},
        )
        entry = state["issues"][str(issue_number)]
        state = write_gate.record_event(
            state,
            "session_failed_escalated",
            {
                "issue_number": issue_number,
                "previous_status": previous_status,
                "reason": "dispatch_blocked_environment",
                "failure_kind": failure_kind,
                "blocking_error": blocking_error,
                "blocked_environment_count": len(blocked_environment_at),
            },
        )
        if save_inline:
            write_gate.save_state(state)
        return LaunchFailureOutcome(state, entry, "escalated", "mechanical")
    # Cap not exceeded: restore without touching the failure counter, and emit a
    # distinct Event so the operator can see the environment conflict before it
    # escalates.
    if clear_stale_escalation:
        clear_escalation(entry)
        clear_escalation_on_issue_prs(state, issue_number)
    if save_inline:
        _seat(state, entry, issue_number, restored_status=restored_status)
    state = write_gate.record_event(  # event-consumer: audit-only -- records a pre-launch environment block (issue #1393) already enforced by the blocked_environment_at counter and the dispatch_blocked_environment escalation; consumed by tests/test_charlie_work.py regression tests.
        state,
        "rework_dispatch_blocked_environment" if rework else "dispatch_blocked_environment",
        {
            "issue_number": issue_number,
            "failure_kind": failure_kind,
            "blocking_error": blocking_error,
            "blocked_environment_count": len(blocked_environment_at),
        },
    )
    if save_inline:
        write_gate.save_state(state)
    return LaunchFailureOutcome(state, entry, restored_status, None)


def _record_generic_failure(
    write_gate: WriteGate,
    state: dict[str, Any],
    entry: dict[str, Any],
    *,
    failure_kind: str | None,
    issue_number: int,
    now: datetime,
    window_minutes: int,
    max_auto_redispatch: int,
    recent_attempts: Callable[[list[str]], list[str]] | None,
    rework: bool,
    save_inline: bool,
    clear_stale_escalation: bool,
    escalation_carries_entry: bool,
    z_stamps: bool,
    restored_status: str,
    cap_reason: str,
) -> LaunchFailureOutcome:
    stamp = now.isoformat()
    if z_stamps:
        stamp = stamp.replace("+00:00", "Z")
    if rework:
        # Every attempt, successful or not, counts against the same windowed
        # ``redispatch_at`` (issue #515); the stored list is the pruned one.
        attempts = _windowed_redispatch_at(entry, window_minutes=window_minutes) + [stamp]
        recent = attempts
    else:
        # Fresh never prunes what it stores (issue #461); only the cap decision
        # looks at the window.
        attempts = list(entry.get("dispatch_failed_at") or [])
        attempts.append(stamp)
        recent = recent_attempts(attempts) if recent_attempts is not None else attempts
        entry["dispatch_failed_at"] = attempts
    # Deterministic launch failures escalate immediately (issue #550); a
    # deterministic judgment failure does too but as ``reason_class="judgment"``
    # (issue #807).
    terminal_failure = failure_kind in DETERMINISTIC_ESCALATION_FAILURE_KINDS
    deterministic_judgment = failure_kind in DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS
    immediate_escalation = terminal_failure or deterministic_judgment
    if immediate_escalation or len(recent) > max_auto_redispatch:
        reason = failure_kind if immediate_escalation else cap_reason
        reason_class = "judgment" if deterministic_judgment else "mechanical"
        state = _escalate_issue(
            state,
            issue_number,
            reason=reason,
            reason_class=reason_class,
            issue_extra=entry
            if escalation_carries_entry
            else {"redispatch_at": attempts, "dispatched_at": None},
        )
        entry = state["issues"][str(issue_number)]
        if save_inline:
            write_gate.save_state(state)
        return LaunchFailureOutcome(state, entry, "escalated", reason_class)
    if clear_stale_escalation:
        clear_escalation(entry)
        clear_escalation_on_issue_prs(state, issue_number)
    if rework:
        entry["redispatch_at"] = attempts
    if save_inline:
        _seat(state, entry, issue_number, restored_status=restored_status)
        write_gate.save_state(state)
    return LaunchFailureOutcome(state, entry, restored_status, None)
