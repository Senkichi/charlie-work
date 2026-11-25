"""What both worker lanes record once a launch has returned (issue #3098).

Fresh ``dispatch`` and remote ``dispatch_rework`` each used to carry their own
copy of the bookkeeping that follows a launch: the ``label_error`` record left
behind when a label transition is not applied, and the blocked-environment
foreign-writer reap. This module owns both, for both lanes; it is step 1 of a
chain that moves the rest of the post-launch handling here (issues #3099,
#3100). It lives at the package root, beside :mod:`charlie_work.worker_launch_gate`,
because every top-level function defined under ``orchestration/`` is
auto-installed on ``OrchestratorApp``.

Shape. The module imports nothing from :mod:`charlie_work.workflow`. Names that
tests replace on ``charlie_work.workflow`` by path (``_try_reap_blocked_foreign_writer``
and friends) stay resolved through ``workflow`` in the lane: the lane hands the
reap in as a zero-argument callable, so those patches keep taking effect.

Lane differences are explicit arguments, never inferred:

- The reap helpers write nothing. They decide, stamp the issue entry they were
  handed and return the Event payload (or ``None`` when no reap happened). Each
  lane emits the Event itself under its own kind, and decides its own status
  change, because the fresh lane records through ``_record_event`` (a ``repo``
  on the Event) and the rework lane through ``append_event`` (no ``repo``) --
  a difference this step preserves.
- :func:`record_label_error` writes through the :class:`WriteGate` it is given
  and nowhere else.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from charlie_work.labels import TransitionOutcome, TransitionResult
from charlie_work.write_gate import WriteGate, require_write_gate


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
    extended with ``now_iso``) and the Event payload is returned for the lane to
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
