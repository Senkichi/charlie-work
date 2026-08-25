from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

STATE_VERSION = 1

# Cross-process lock timeout (seconds) — best-effort to prevent wedging
_LOCK_TIMEOUT_SECONDS = 30

# Retry for transient read errors (e.g. Windows sharing violations) before
# treating the file as unrecoverable.
_LOAD_RETRY_ATTEMPTS = 3
_LOAD_RETRY_DELAY_SECONDS = 0.1

# Retry for transient write errors on the atomic replace (issue #1062). On
# Windows, ``os.replace()`` onto a target another process currently holds open
# (a lock-free ``load_state`` reader, ``charlie doctor``, a dashboard render)
# raises ``PermissionError`` [WinError 5]. The failure is transient and
# non-destructive -- the previous valid file is intact -- so retry with backoff
# before surfacing, mirroring the reader-side knobs above.
_SAVE_RETRY_ATTEMPTS = 3
_SAVE_RETRY_DELAY_SECONDS = 0.1

# Stale claim timeout (minutes) — claims older than this are re-dispatchable
# to prevent crashed phase-2 from wedging issues
_STALE_CLAIM_TIMEOUT_MINUTES = 30

# Default event ring cap. OrchestratorApp sets EVENT_RING_SIZE from
# RuntimeConfig at startup so the bound is config-driven (issue #525).
DEFAULT_EVENT_RING_SIZE = 2000
EVENT_RING_SIZE = DEFAULT_EVENT_RING_SIZE

# Reviewer-specific stale claim timeout (minutes). Session-limit kills are
# detectable within seconds (the reviewer dies and prints the limit message to
# its log), so the 30-minute worker timeout is far too long for review
# dispatches: it extends the hot-redispatch loop cycle unnecessarily. 5
# minutes is ample for a reviewer that started successfully but died from a
# throttle, while still avoiding thrash on flaky launch paths.
_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES = 5

# Orphan backstop for dead-reviewer claims at the CLAIM stage (issue #571).
# A dispatched claim whose reviewer has died must be dispositioned by the
# stalled-review sweep first (throttle classification, probe-failure count,
# sidecar reap, events) — the claim stage racing it on the same 5-minute
# timeout measured later in the pass produced a deterministic livelock:
# every relaunch reset the clock just after the sweep looked, so the sweep
# never saw a stale claim and quota backoff never engaged. 3x the stale
# timeout gives the sweep three windows of first refusal while still
# self-healing true orphans (e.g. a crash before the sidecar was written,
# which the worker-iterating sweep cannot see).
_REVIEW_DEAD_CLAIM_BACKSTOP_TIMEOUT_MINUTES = _REVIEW_STALE_CLAIM_TIMEOUT_MINUTES * 3

# Every literal ever assigned to `issues[n]["status"]` across the
# orchestrator's dispatch -> review -> rework -> merge lifecycle (workflow.py).
# reconcile.py's status-normalization sweep treats any issue record whose
# status is missing or falls outside this set as drift and recomputes it from
# ground truth. Kept here rather than in reconcile.py so any future writer of
# issue status has a single, importable source of truth to validate against --
# the same reasoning that already put ``without_review_dispatch_claim`` here
# instead of scattered across callers.
VALID_ISSUE_STATUSES: frozenset[str] = frozenset(
    {
        "dispatched",
        "dispatch_pending",
        # manual-adapter dispatch: manifest written, worker not yet confirmed
        "manifest_written",
        # non-terminal dispatch failure awaiting windowed redispatch (#461);
        # omitting it here would let the normalization sweep strip the status
        # and re-expose the issue as a fresh dispatch candidate past the cap
        "dispatch_failed",
        "reviewing",
        # Issue #955: PASSIVE_OPEN_STATUS's own value, distinct from the
        # active "reviewing" above -- see that constant's docstring.
        "open_passive",
        "rework_requested",
        "approved",
        "blocked",
        "escalated",
        "closed",
    }
)

# "closed" is the one VALID_ISSUE_STATUSES member the orchestrator does not
# own: it is mirrored from GitHub's issue state, and GitHub can invalidate it
# at any time via a reopen. Every other member is written exclusively by the
# orchestrator's own dispatch -> review -> rework -> merge transitions, so
# reconcile's status-normalization sweep is right to treat them as
# self-validating -- but doing the same for "closed" turns a GitHub reopen
# into a one-way gate: state can enter "closed" from GitHub, but can never
# leave it, because the sweep never looks again (issue #789). Declaring the
# split here means the sweep derives its skip-set from this frozenset
# subtraction instead of a `!= "closed"` special case at the call site, so a
# future externally-derived status is covered by construction.
EXTERNALLY_DERIVED_ISSUE_STATUSES: frozenset[str] = frozenset({"closed"})
ORCHESTRATOR_OWNED_ISSUE_STATUSES: frozenset[str] = (
    VALID_ISSUE_STATUSES - EXTERNALLY_DERIVED_ISSUE_STATUSES
)

# Issue #955: this used to be the literal string "reviewing" -- the same
# value ``review()`` writes (guarded by ``review_dispatch.enabled``, see
# workflow.py's two ``dispatch_disabled`` call sites) to mean "a fresh review
# packet was generated, a reviewer is expected". Sharing one string across
# both meanings meant a reader could not tell "a reviewer is coming" from
# "this record is just open, nothing is tracked yet" -- reconcile.py's
# unconditional ``pr_status_normalized`` write landed PRs in the exact status
# the #487 stalled-review sweep (workflow.py) keys on to detect "packet
# generated but never dispatched", producing one spurious
# ``review_dispatch_stalled`` event per PR whenever review dispatch is
# disabled. See the issue for the full trace.
#
# PASSIVE_OPEN_STATUS is now a distinct value, used only where the record
# should read "open, no reviewer implied" -- reconcile.py's self-heal and
# status-normalization sweeps (for both issue and PR records), and
# workflow.py's ``unescalate``/mechanical-deescalation/orphaned-worker-
# opened-a-PR recovery paths. None of these fire the ``review_started`` label
# transition or claim a reviewer is coming; they exist so a drift-repair or
# reset pass never resurrects an issue into "rework_requested" (which would
# trigger a fresh worker dispatch) purely by fixing a label or a corrupt
# status field.
#
# It must stay a member of ``reconcile.ACTIVE_STATE_STATUSES`` alongside the
# real "reviewing" status: both are "this entry is in the open pipeline"
# states that the closed-unmerged-PR and issue-closed-on-GitHub repair sweeps
# must keep converging to a terminal status. Dropping it from that set to
# "fix the naming" would silently disable those repairs for every
# passively-open entry.
PASSIVE_OPEN_STATUS = "open_passive"

# Issue #783: every transition into the ``human_needed`` label must record
# WHY, durably and atomically with the label change, so an automated
# de-escalation sweep can tell a process failure from a substantive judgment
# call. Escalating on a dead worker process when the PR artifact itself is
# fine is a category error -- these two classes exist to stop treating that
# the same as a human product/security decision that must stay terminal.
#
#   "mechanical" -- a process/infrastructure failure: dead worker session,
#     redispatch/rework-cycle cap exhausted, a stalled worker, or a
#     janitor-detected merge conflict/CI failure past its retry cap.
#     Self-clearing: workflow.py's ``_maybe_deescalate_mechanical`` sweep may
#     re-evaluate and auto-clear it once the PR is mergeable and janitor_ok.
#   "judgment" -- a human product or security decision, an unimplementable
#     acceptance criterion, or a reviewer's explicit "blocked" verdict. Stays
#     terminal; only a human running ``charlie unescalate`` may clear it.
#
# A legacy escalation recorded with no ``reason_class`` at all may be
# backfilled by ``workflow._maybe_deescalate_mechanical`` from the most
# recent escalation-transition event in ``events.db``, but only when the
# event kind unambiguously denotes a process failure. Ambiguous or
# deliberately-preserved kinds stay fail-closed: ``reason_class`` remains
# absent and the issue stays terminal.
ESCALATION_REASON_CLASSES: frozenset[str] = frozenset({"mechanical", "judgment"})

# Issue #797: legacy escalations may lack ``reason_class`` because the field
# was added later. The backfill derives the class from the escalation event
# kind. Only kinds that unambiguously indicate a process failure map to
# ``"mechanical"``. Kinds in ``DELIBERATELY_UNCLASSIFIED_ESCALATION_EVENT_KINDS``
# are too ambiguous (or are intentionally preserved forensic records) and
# must stay unclassified, so the backfill leaves the issue terminal.
ESCALATION_REASON_CLASS_BY_EVENT_KIND: Mapping[str, str] = MappingProxyType(
    {
        # A rework worker's session dying, or a redispatch/no-op-rework cap
        # being exceeded, is a pure process/infrastructure failure.
        "session_failed_escalated": "mechanical",
        # The review-dispatch attempt cap is an infrastructure-driven limit.
        "review_dispatch_escalated": "mechanical",
        # Issue #841: an infra-cancelled required check (self-hosted runner
        # timeout-minutes kill) exhausting its auto-rerun attempt cap is a
        # pure process/infrastructure limit, unambiguous like the other
        # attempt-cap kinds above -- no code fix or human judgment call is
        # involved, only a retry budget running out.
        "infra_rerun_escalated": "mechanical",
        # Issue #1010: the pre-flight cross-repo gate escalated an issue
        # because every file path it referenced was absent from the target
        # repo. This is a mechanical dispatch-time determination (a path
        # existence check), not a human judgment call -- the worker cannot
        # fix it, and a human re-triages the target repo.
        "dispatch_cross_repo_escalated": "mechanical",
    }
)
DELIBERATELY_UNCLASSIFIED_ESCALATION_EVENT_KINDS: frozenset[str] = frozenset(
    {
        # Issue #662: a deliberately-preserved forensic record; the kind alone
        # cannot distinguish a normal process failure from a record that must
        # stay terminal.
        "janitor_rework_escalated",
        # Can carry either a human ``blocked`` verdict (judgment) or a
        # request-changes/unparseable report (mechanical); the kind alone is
        # ambiguous.
        "rescue_review_escalated",
        # Records review decisions including approved, request_changes, and
        # blocked; the event kind alone does not identify an escalation.
        "record_review",
        # Diagnostics-only: emitted either (a) while re-running the janitor
        # for visibility on a PR/issue that is ALREADY escalated (this
        # occurrence's ``escalated: True`` payload key documents the
        # ambient status, it does not cause the transition -- see
        # ``review()``'s "Escalation is terminal" branch), or (b) as the
        # ordinary janitor-gate block, which sets status "janitor_blocked",
        # never "escalated". The kind alone never identifies an escalation
        # transition.
        #
        # Membership here is inert for `_backfill_missing_reason_classes`'s
        # per-issue `query_events(issue_number=...)` lookup: neither payload
        # above carries an `issue_number`/`issue`/`issue_numbers`/`issues`
        # key, so `_extract_payload_refs` (instrumentation.py) always leaves
        # the indexed `issue_number` column NULL for this kind, and SQL
        # `issue_number = N` never matches NULL -- a `janitor_gate` row can
        # never be selected as the "latest" event for any issue. Mapping it
        # to `None` here is therefore unreachable in practice, not merely
        # unmapped by omission.
        #
        # Discovery of this kind by
        # ``_escalation_event_kinds_from_workflow()`` in
        # ``test_deescalation.py`` is contingent on `review()` containing a
        # call to `escalation_reason_class(...)` (currently true via the
        # `infra_rerun_escalated` block). If a future refactor moves that
        # call out of `review()`, this kind stops being discovered and this
        # entry becomes dormant -- harmless, but worth knowing if this
        # comment is ever found to be describing a kind the test no longer
        # flags.
        "janitor_gate",
        # Same shape as janitor_gate immediately above: diagnostics-only,
        # emitted either while re-running janitor diagnostics for visibility
        # on an already-escalated PR (that occurrence's ``escalated: True``
        # payload key documents the ambient status, it does not cause the
        # transition) or from the ordinary non-escalated janitor-gate block,
        # which never sets status "escalated". The kind alone never
        # identifies an escalation transition.
        "ci_run_never_created",
        # Issue #1131: a rework-label skip diagnostic, not an escalation
        # transition. Emitted inside ``record_review`` (which does perform
        # escalation elsewhere), so the AST-based discovery in
        # ``test_deescalation.py`` picks it up -- but the kind alone never
        # identifies an escalation: it fires when rework routing is
        # suppressed for a CLOSED issue, the opposite of an escalation.
        "rework_label_skipped_issue_closed",
    }
)


def escalation_reason_class(reason_class: str) -> str:
    """Validate an escalation ``reason_class`` value before it is persisted.

    Raises ``ValueError`` on anything other than ``"mechanical"`` or
    ``"judgment"`` so a typo at a call site fails loudly at write time
    instead of silently producing an escalation the de-escalation sweep can
    never recognize as mechanical (a safe-but-pointless failure direction).
    """
    if reason_class not in ESCALATION_REASON_CLASSES:
        raise ValueError(f"invalid escalation reason_class: {reason_class!r}")
    return reason_class


def clear_escalation(entry: dict[str, Any]) -> dict[str, Any]:
    """Remove the paired escalation fields from ``entry``.

    Safe to call on any dict: missing keys are ignored. This is the
    single-point inverse of the escalation write path and must be used on
    every code path that clears an escalation, so ``reason_class`` can never
    survive after its ``escalation_reason`` is removed.

    Issue #1461: also clears ``escalation_reasons_seen`` so a de-escalated
    issue gets a genuinely fresh escalation history on re-escalation --
    without this, a per-lane dedup guard that checks membership in the list
    would permanently suppress the lane that previously escalated, even
    after the operator or auto-sweep cleared the escalation.
    """
    entry.pop("escalation_reason", None)
    entry.pop("reason_class", None)
    entry.pop("escalation_reasons_seen", None)
    return entry


def clear_escalation_on_issue_prs(state: dict[str, Any], issue_number: int) -> bool:
    """Mirror-clear escalation fields on every PR record linked to an issue.

    The escalation write path (``_escalate_issue``) writes
    ``escalation_reason`` and ``escalation_reasons_seen`` to *both* the issue
    record and the PR record.  Before this helper existed, every
    ``clear_escalation`` call site cleared the issue-side fields only, leaving
    a stale ``escalation_reason`` on the PR record.  The downstream rework
    router (``_route_janitor_gate_failure_to_rework``) short-circuits on
    ``existing_pr_state.get("escalation_reasons_seen")`` (issue #1461: was
    ``escalation_reason``), so an issue whose escalation was "cleared" still
    routed nowhere -- a visible stuck state converted into silence (issue
    #1093).

    This helper is the single-point mirror-clear: call it at every
    ``clear_escalation`` site that has a resolved issue number, and the PR
    records stay in sync with the issue record.  ``reason_class`` is also
    popped (a no-op on PR records, which never carry it, but harmless and
    keeps the pair symmetric with ``clear_escalation``).

    Mutates PR entries in place inside ``state["prs"]``.  Returns ``True`` if
    at least one PR record was found and cleared.
    """
    cleared_any = False
    for key, entry in state.get("prs", {}).items():
        if isinstance(entry, dict) and entry.get("issue_number") == issue_number and key.isdigit():
            clear_escalation(entry)
            cleared_any = True
    return cleared_any


logger = logging.getLogger(__name__)


class StateLockBusy(RuntimeError):
    """Raised when the advisory state lock cannot be acquired within its budget.

    A state writer that cannot acquire the lock must fail that unit of work as
    a value (skip + event log), never write unlocked.
    """


# Intra-process serialization for state_lock.
#
# The file lock below (msvcrt.locking / fcntl.flock) serializes across
# PROCESSES, but byte-range file locks are owned by the process, not the
# thread — two threads in the SAME process are not serialized by it and both
# enter the read-modify-write section concurrently. On Windows their atomic
# ``tmp.replace(state.json)`` calls then collide (destination held open by the
# other thread's read) and raise ``PermissionError``; on any platform the
# concurrent load→save races lose updates (issue #16).
#
# A per-path threading.Lock, acquired before the file lock, restores
# deterministic intra-process serialization. Keyed by normalized absolute path
# so distinct Path objects for the same file share one lock. The registry
# itself is guarded by a plain lock; entries are created on demand and never
# removed (one small Lock per distinct state path for the process lifetime).
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock_for(path: Path) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(path))
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[key] = lock
        return lock


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def without_review_dispatch_claim(pr_state: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a PR state entry with all review-dispatch claim fields cleared.

    This is the single helper for finalizing a PR whose GitHub lifecycle has
    moved on (merged or closed externally) or whose review claim is otherwise
    moot. It never mutates the input dict.
    """
    return {
        **pr_state,
        "review_dispatch_status": None,
        "review_dispatch_pending_at": None,
        "review_dispatched_at": None,
        "review_dispatch_failed_at": None,
        "reviewer_pid": None,
        "reviewer_process_start_time": None,
    }


def _to_float(value: Any) -> float | None:
    """Coerce a JSON-deserialized value to a float, or return None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _canonical_started_at(started_at: Any, process_start_time: Any | None = None) -> str:
    """Coerce ``started_at`` to a canonical ISO-8601 UTC string (Z, no microseconds).

    Accepts ISO-8601 strings (with or without timezone, with ``Z`` or ``+HH:MM``)
    and numeric Unix timestamps. If ``started_at`` is missing or unparseable, falls
    back to ``process_start_time`` (a Unix timestamp). Raises ``ValueError`` if no
    usable timestamp can be produced.
    """
    if started_at is None:
        started_at_str = ""
    else:
        started_at_str = str(started_at).strip()
    if started_at_str in {"", "None", "null"}:
        started_at_str = ""

    if started_at_str:
        try:
            parsed = datetime.fromisoformat(started_at_str)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            parsed = parsed.astimezone(UTC)
            return parsed.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        except (ValueError, TypeError):
            pass

    # Fall back to the process start time, or a numeric started_at string.
    fallback_ts = _to_float(process_start_time)
    if fallback_ts is None and started_at_str:
        fallback_ts = _to_float(started_at_str)
    if fallback_ts is not None:
        try:
            parsed = datetime.fromtimestamp(fallback_ts, tz=UTC)
            return parsed.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        except (ValueError, OSError, OverflowError):
            pass

    raise ValueError(
        f"started_at must be a valid ISO-8601 timestamp or numeric Unix timestamp; "
        f"got {started_at!r}"
    )


def is_claim_stale(
    claim_timestamp: str | None,
    *,
    timeout_minutes: int = _STALE_CLAIM_TIMEOUT_MINUTES,
    now: datetime | None = None,
) -> bool:
    """Check if a dispatch_pending claim is stale and should be re-dispatchable.

    A claim is stale if it's older than ``timeout_minutes``.
    This prevents crashed phase-2 processes from wedging issues permanently.

    ``now`` is the injectable clock (issue #828): defaults to
    ``datetime.now(UTC)`` when not supplied, so production behavior is
    byte-identical. A caller evaluating several claims against one shared
    instant (e.g. a single sweep pass) should sample ``now`` once and pass
    the same value through, instead of letting each check independently
    race the wall clock.
    """
    if not claim_timestamp:
        return False
    try:
        claim_time = datetime.fromisoformat(claim_timestamp.replace("Z", "+00:00"))
        resolved_now = now if now is not None else datetime.now(UTC)
        age = resolved_now - claim_time
        return age > timedelta(minutes=timeout_minutes)
    except (ValueError, TypeError):
        # Malformed timestamp — treat as stale to be safe
        return True


def _operator_claim_timestamp(entry: Any) -> str | None:
    """Return the operator_claimed_at timestamp for an issue entry, if any."""
    if not isinstance(entry, dict):
        return None
    return entry.get("operator_claimed_at") or None


def is_operator_claimed(entry: Any) -> bool:
    """Return True when an issue entry carries a live operator claim.

    Operator claims are intentionally not auto-expired: only an explicit
    ``--release`` removes the claim. This prevents an operator from being
    silently displaced while working in a worktree.
    """
    return _operator_claim_timestamp(entry) is not None


def operator_claimed_issues(data: dict[str, Any]) -> set[int]:
    """Return the set of issue numbers currently under an operator claim."""
    claimed: set[int] = set()
    for issue_number_str, entry in data.get("issues", {}).items():
        if is_operator_claimed(entry):
            try:
                claimed.add(int(issue_number_str))
            except (ValueError, TypeError):
                continue
    return claimed


def stale_operator_claims(
    data: dict[str, Any], threshold_minutes: int = _STALE_CLAIM_TIMEOUT_MINUTES
) -> set[int]:
    """Return issue numbers whose operator claim is older than ``threshold_minutes``.

    Used for digest warnings; stale claims still block dispatch until released.
    """
    now = datetime.now(UTC)
    stale: set[int] = set()
    for issue_number_str, entry in data.get("issues", {}).items():
        timestamp = _operator_claim_timestamp(entry)
        if not timestamp:
            continue
        try:
            claim_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if (now - claim_time) > timedelta(minutes=threshold_minutes):
                stale.add(int(issue_number_str))
        except (ValueError, TypeError):
            # Malformed timestamp — treat as stale to be safe
            stale.add(int(issue_number_str))
    return stale


def set_operator_claimed(
    data: dict[str, Any], issue_number: int, timestamp: str | None = None
) -> dict[str, Any]:
    """Return a new state dict with ``operator_claimed_at`` set for ``issue_number``.

    Does not mutate ``data``.
    """
    timestamp = timestamp or utc_now()
    issue_key = str(issue_number)
    entry = {
        **data.get("issues", {}).get(issue_key, {}),
        "number": issue_number,
        "operator_claimed_at": timestamp,
    }
    return {**data, "issues": {**data.get("issues", {}), issue_key: entry}}


def release_operator_claimed(data: dict[str, Any], issue_number: int) -> dict[str, Any]:
    """Return a new state dict with the operator claim removed.

    Does not mutate ``data``. Removes the issue entry if it becomes empty
    (after preserving ``number``).
    """
    issue_key = str(issue_number)
    entry = {
        k: v
        for k, v in data.get("issues", {}).get(issue_key, {}).items()
        if k != "operator_claimed_at"
    }
    if not entry:
        issues = {k: v for k, v in data.get("issues", {}).items() if k != issue_key}
    else:
        issues = {**data.get("issues", {}), issue_key: entry}
    return {**data, "issues": issues}


@contextmanager
def advisory_file_lock(path: Path):
    """Cross-process advisory lock for a JSON file's read-modify-write cycle.

    This is the generic primitive behind ``state_lock``. It serializes the
    load→modify→save cycle on ANY atomic-JSON state file (``state.json``,
    ``api-budget.json``, …) so concurrent writers cannot lose updates: the
    file lock serializes across PROCESSES and a per-path ``threading.Lock``
    serializes concurrent THREADS in this process (byte-range file locks are
    owned by the process, not the thread — see ``_thread_lock_for``).

    Uses platform-specific file locking (Windows: ``msvcrt.locking``, POSIX:
    ``fcntl.flock``) on a ``<path>.lock`` lockfile alongside the target. The
    lock is advisory and time-bounded (``_LOCK_TIMEOUT_SECONDS``) to prevent
    wedging on stale locks.

    Deterministic: if the lock cannot be acquired within the timeout, the
    context manager raises ``StateLockBusy``. A writer that cannot acquire the
    lock must fail that unit of work as a value (skip + event log), never write
    unlocked.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_file = None
    acquired = False

    thread_lock = _thread_lock_for(path)
    thread_lock.acquire()
    try:
        # Create the lock file if needed. touch() leaves it at 0 bytes, which
        # is fine: msvcrt.locking(..., LK_NBLCK, 1) succeeds on a 0-byte file
        # on the deployed runtime (Python 3.13.5, Windows 11) — probed in
        # #324/#328, which removed the same write-1-byte guards from
        # file_lock.py as dead code.
        lock_path.touch()

        if sys.platform == "win32":
            import msvcrt

            lock_file = lock_path.open("r+b", encoding=None)
            # msvcrt.locking mode: 0 = lock, 1 = unlock
            # LK_NBLCK = non-blocking lock, LK_LOCK = blocking lock
            # We use a retry loop with timeout for bounded waiting
            import time

            start = time.time()
            while time.time() - start < _LOCK_TIMEOUT_SECONDS:
                try:
                    # Try non-blocking lock first
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except OSError:
                    # Lock held, wait and retry
                    time.sleep(0.1)
            else:
                # Timeout — the lock is held by another writer. Fail this unit
                # of work as a value rather than degrading integrity.
                logger.warning(
                    f"Failed to acquire lock on {lock_path} after {_LOCK_TIMEOUT_SECONDS}s"
                )
                raise StateLockBusy(
                    f"Could not acquire state lock at {lock_path} within {_LOCK_TIMEOUT_SECONDS}s"
                )
        else:
            import fcntl
            import time

            lock_file = lock_path.open("r+b", encoding=None)
            start = time.time()
            while time.time() - start < _LOCK_TIMEOUT_SECONDS:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (IOError, BlockingIOError):
                    # Lock held, wait and retry
                    time.sleep(0.1)
            else:
                # Timeout — the lock is held by another writer. Fail this unit
                # of work as a value rather than degrading integrity.
                logger.warning(
                    f"Failed to acquire lock on {lock_path} after {_LOCK_TIMEOUT_SECONDS}s"
                )
                raise StateLockBusy(
                    f"Could not acquire state lock at {lock_path} within {_LOCK_TIMEOUT_SECONDS}s"
                )

        yield
    finally:
        # Close whenever the handle was opened, regardless of whether the
        # lock was acquired — on the timeout path the handle was still opened
        # above and must not leak. Unlock only when acquired=True (nothing to
        # unlock otherwise). The two operations are independent failure modes:
        # an unlock raising OSError must not skip the close.
        if lock_file is not None:
            if acquired:
                try:
                    if sys.platform == "win32":
                        import msvcrt

                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except OSError:
                    # Best-effort unlock — ignore failures
                    pass
            try:
                lock_file.close()
            except OSError:
                # Best-effort close — ignore failures
                pass
        # Release the intra-process thread lock last, after the file handle is
        # closed, so the next thread never observes a half-released critical
        # section. Always paired with the acquire() above the try.
        thread_lock.release()


@contextmanager
def state_lock(state_path: Path):
    """Cross-process advisory lock for state.json read-modify-write cycles.

    Thin wrapper over ``advisory_file_lock`` (the generic primitive) kept under
    the original name so the many existing call sites are unchanged. See
    ``advisory_file_lock`` for the locking semantics, timeout, and the
    intra-process thread serialization rationale.
    """
    with advisory_file_lock(state_path):
        yield


def empty_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "generated_at": utc_now(),
        "issues": {},
        "prs": {},
        "events": [],
        "throttled_until": None,  # ISO timestamp when provider throttle cooldown ends
        # Issue #1001: durable once-only escalation marker for the worker
        # GitHub token gate. A missing token is a standing condition; the gate
        # must not emit a worker_token_missing event every loop pass.
        # fleet_dispatch.fleet_loop constructs a fresh OrchestratorApp per repo
        # per pass, so an instance-level flag alone resets every pass and
        # re-escalates indefinitely. This marker persists across reconstruction
        # and is cleared when the condition resolves (token added), so a future
        # regression re-escalates. Lives in state.json per the "state lives in
        # GitHub labels + state.json" invariant.
        "worker_token_escalated": False,
    }


def _quarantine_state(path: Path, exc: Exception) -> None:
    """Rename an unparseable state file for forensics and log a loud signal.

    The dispatch/loop path calls ``load_state`` frequently; emitting a
    top-level error here makes a silent state wipe visible in logs.
    """
    quarantine = path.with_name(f"{path.name}.corrupt-{utc_now().replace(':', '')}")
    logger.error(
        "State file %s is unrecoverable (%s: %s); quarantining to %s",
        path,
        type(exc).__name__,
        exc,
        quarantine,
    )
    try:
        path.replace(quarantine)
    except OSError as move_err:
        logger.error("Failed to quarantine state file %s: %s", path, move_err)


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_state()

    data: Any = None
    for attempt in range(_LOAD_RETRY_ATTEMPTS):
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            # Genuine JSON corruption (truncated files, etc.) — quarantine.
            _quarantine_state(path, exc)
            return empty_state()
        except (LookupError, ValueError) as exc:
            # Decoding-level corruption (e.g. UTF-16LE+BOM, unknown encoding).
            # A wrong-encoding state file is not a transient read error.
            _quarantine_state(path, exc)
            return empty_state()
        except OSError as exc:
            # Sharing/permission violations on Windows are often transient.
            # Retry before falling back to quarantine.
            if attempt < _LOAD_RETRY_ATTEMPTS - 1:
                time.sleep(_LOAD_RETRY_DELAY_SECONDS)
                continue
            _quarantine_state(path, exc)
            return empty_state()
        else:
            break

    if not isinstance(data, dict):
        return empty_state()
    data.setdefault("version", STATE_VERSION)
    data.setdefault("generated_at", utc_now())
    data.setdefault("issues", {})
    data.setdefault("prs", {})
    data.setdefault("events", [])
    data.setdefault("throttled_until", None)  # Backward compatibility
    data.setdefault("worker_token_escalated", False)  # Issue #1001
    return data


def save_state(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    """Persist a fresh copy of ``data`` without mutating the caller's dict."""
    to_save = {**data, "generated_at": utc_now()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(to_save, handle, indent=2, sort_keys=True)
        handle.write("\n")
    # On Windows, ``replace()`` onto a target another process currently holds
    # open (a lock-free ``load_state`` reader, ``charlie doctor``, a dashboard
    # render) raises ``PermissionError`` [WinError 5]. The failure is transient
    # and non-destructive -- the previous valid file is intact -- so retry with
    # backoff before surfacing, mirroring ``load_state``'s reader-side retry.
    # ``PermissionError`` is caught specifically so the final message can name
    # the condition; a bare "Access is denied" sends an operator hunting for an
    # admin shell (issue #1062).
    for attempt in range(_SAVE_RETRY_ATTEMPTS):
        try:
            tmp_path.replace(path)
            break
        except PermissionError as exc:
            if attempt < _SAVE_RETRY_ATTEMPTS - 1:
                time.sleep(_SAVE_RETRY_DELAY_SECONDS)
                continue
            raise PermissionError(
                f"atomic replace of {path} failed after {_SAVE_RETRY_ATTEMPTS} "
                f"attempts: {exc}. This is usually a transient Windows sharing "
                f"violation (another process holds the file open); the previous "
                f"state file is intact."
            ) from exc
    return to_save


def load_state_locked(path: Path) -> dict[str, Any]:
    """Load a state snapshot while holding the advisory lock.

    This is the single point of enforcement for read-only ``load_state`` calls
    outside an explicit ``state_lock`` context. Callers receive a fresh snapshot
    and must not mutate it without re-acquiring the lock and saving explicitly.
    """
    with state_lock(path):
        return load_state(path)


def append_event(
    data: dict[str, Any],
    kind: str,
    payload: dict[str, Any],
    max_size: int | None = None,
    *,
    state_path: Path | None = None,
    repo: str | None = None,
    level: str | None = None,
) -> dict[str, Any]:
    """Return a new state dict with the event appended; do not mutate ``data``.

    ``max_size`` defaults to the module-level ``EVENT_RING_SIZE``, which
    OrchestratorApp sets from ``RuntimeConfig.event_ring_size`` at startup.
    Callers (including tests) may pass an explicit cap to pin the truncation
    behavior they are validating.

    When ``state_path`` is provided, the event is also written to the
    unlimited append-only ``events.db`` SQLite log alongside ``state.json``.
    This dual-write preserves the complete audit history beyond the bounded
    convenience cap in ``state.json``'s ``events`` array. The write is
    best-effort — instrumentation never breaks the core workflow.

    ``level`` is forwarded to ``log_event`` when ``state_path`` is given so
    emit sites can declare a level explicitly rather than relying on the
    central registry.
    """
    if max_size is None:
        max_size = EVENT_RING_SIZE
    events = list(data.get("events", []))
    events.append({"at": utc_now(), "kind": kind, "payload": payload})
    if len(events) > max_size:
        events = events[-max_size:]
    if state_path is not None:
        from .instrumentation import log_event

        log_event(state_path, kind, payload, repo=repo, level=level)
    return {**data, "events": events}


def is_throttled(data: dict[str, Any]) -> bool:
    """Check if the orchestrator is currently in a provider throttle cooldown window.

    Returns True if now < throttled_until, False otherwise.
    """
    throttled_until = data.get("throttled_until")
    if not throttled_until:
        return False
    try:
        throttle_time = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
        return datetime.now(UTC) < throttle_time
    except (ValueError, TypeError):
        # Malformed timestamp — treat as not throttled to be safe
        return False


def set_throttled_until(
    data: dict[str, Any],
    throttled_until: str,
    *,
    reason: str | None = None,
    adapter_kind: str | None = None,
) -> dict[str, Any]:
    """Set the provider throttle cooldown window.

    ``reason`` (a ``_classify_session_failure`` failure_kind -- e.g.
    "quota_exhausted", "provider_auth", "rate_limited") and ``adapter_kind``
    (the adapter whose session hit the throttle) are recorded alongside the
    cooldown so a later quota-probe decision (``clear_quota_throttles``) can
    tell a self-healing rate limit from a dead credential that must not be
    cleared early, and can tell whether an ambient-CLI probe actually
    exercises the provider that was throttled. Both default to None so
    call sites that have not been updated to pass them keep working; a
    throttle with an unset reason/adapter_kind is treated as
    claude-code-shaped (the common case) by ``clear_quota_throttles``.

    Returns a new state dict with throttled_until set; does not mutate ``data``.
    """
    return {
        **data,
        "throttled_until": throttled_until,
        "throttle_reason": reason,
        "throttle_adapter_kind": adapter_kind,
    }


def _reviewer_quota(data: dict[str, Any]) -> dict[str, Any]:
    """Return the reviewer quota sub-dict from ``data``.

    Ensures a mutable copy is returned so callers can build new state without
    mutating the original ``data``.
    """
    quota = data.get("reviewer_quota")
    if not isinstance(quota, dict):
        return {}
    return dict(quota)


def is_reviewer_quota_exhausted(data: dict[str, Any]) -> bool:
    """Check if reviewer quota is currently exhausted.

    True when ``reviewer_quota.throttled_until`` is a future timestamp.
    Malformed timestamps are treated as not exhausted.
    """
    throttled_until = _reviewer_quota(data).get("throttled_until")
    if not throttled_until:
        return False
    try:
        throttle_time = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
        return datetime.now(UTC) < throttle_time
    except (ValueError, TypeError):
        return False


def is_reviewer_probe_ready(data: dict[str, Any]) -> bool:
    """Check if enough time has passed to attempt a reviewer quota probe.

    True when ``reviewer_quota.probe_after`` is absent or in the past.
    Malformed timestamps are treated as ready to avoid wedging dispatch.
    """
    probe_after = _reviewer_quota(data).get("probe_after")
    if not probe_after:
        return True
    try:
        probe_time = datetime.fromisoformat(probe_after.replace("Z", "+00:00"))
        return datetime.now(UTC) >= probe_time
    except (ValueError, TypeError):
        return True


def set_reviewer_quota_exhausted(
    data: dict[str, Any], *, throttled_until: str, probe_after: str
) -> dict[str, Any]:
    """Set reviewer quota exhaustion and the next probe timestamp.

    Returns a new state dict; does not mutate ``data``.
    """
    quota = _reviewer_quota(data)
    quota["throttled_until"] = throttled_until
    quota["probe_after"] = probe_after
    return {**data, "reviewer_quota": quota}


def mark_reviewer_quota_alerted(data: dict[str, Any]) -> dict[str, Any]:
    """Record that the current quota-exhaustion episode has been alerted.

    One attention digest per exhaustion episode: the marker is cleared with
    the rest of the quota record when the probe succeeds, so a later episode
    alerts again. Returns a new state dict; does not mutate ``data``.
    """
    quota = _reviewer_quota(data)
    quota["alerted_at"] = utc_now()
    return {**data, "reviewer_quota": quota}


def clear_reviewer_quota(data: dict[str, Any]) -> dict[str, Any]:
    """Clear reviewer quota exhaustion state.

    Returns a new state dict; does not mutate ``data``. The
    ``last_probe_cleared_at`` recovery marker (written by
    ``clear_quota_throttles`` and by the verdict-reap recovery path in
    ``workflow.dispatch_reviews``) is deliberately preserved across clears so
    the dead-reviewer reap sweep can compare it against a dead session's death
    time across exhaustion episodes (issue #662).
    """
    quota = _reviewer_quota(data)
    if not quota:
        return data
    quota.pop("throttled_until", None)
    quota.pop("probe_after", None)
    quota.pop("alerted_at", None)
    # Issue #612: the parsed provider reset time is per-episode; clear it so
    # a stale value does not linger after the quota window is proven open.
    quota.pop("reset_at", None)
    return {**data, "reviewer_quota": quota}


def reviewer_quota_last_probe_cleared_at(data: dict[str, Any]) -> str | None:
    """Return the timestamp of the last green probe or verdict that cleared a
    throttle.

    Written by ``clear_quota_throttles`` whenever a green ambient-CLI probe
    clears a claude-code-shaped throttle, and by the verdict-reap recovery
    path in ``workflow.dispatch_reviews`` whenever a dead reviewer's verdict
    proves the provider window is open. Consumed by the dead-reviewer reap
    sweep (``_detect_and_handle_stalled_reviews``) to suppress re-poisoning
    ``reviewer_quota`` from a throttle signature frozen in a dead session's
    log tail when a recovery has already happened after that session died
    (issue #662). None when no green probe/verdict has cleared the quota
    since the last exhaustion episode (or ever).
    """
    return _reviewer_quota(data).get("last_probe_cleared_at")


def defer_reviewer_probe_after(data: dict[str, Any], probe_after: str) -> dict[str, Any]:
    """Bump ``reviewer_quota.probe_after`` to at least ``probe_after``.

    Called from the flat-interval quota probe's red branch so that
    ``dispatch_reviews``'s own ``probe_mode`` gate (which reads
    ``is_reviewer_probe_ready``) defers at least until the cheap probe's
    next scheduled attempt, instead of independently launching a full
    reviewer session into a window the cheap probe just confirmed is still
    closed (issue #663). Without this, the two probing mechanisms run on
    independent schedules and don't share failure-path state: a red flat
    probe only re-armed ``quota_probe.next_probe_at``, leaving
    ``reviewer_quota.probe_after`` in the past, so ``dispatch_reviews``
    could launch a real (non-Haiku) reviewer session as its own probe of
    the same still-closed window -- wasting a bounded (``dispatch_limit=1``)
    but real reviewer dispatch.

    Only acts when the reviewer quota is currently exhausted: writing
    ``probe_after`` on a non-exhausted quota would leave stale state that
    ``clear_reviewer_quota`` would later clean up, but there is no reason
    to set it in the first place. Never moves ``probe_after`` earlier --
    the reviewer quota's own exponential backoff may have already pushed
    it further out than the flat probe's interval, and shortening it would
    make ``dispatch_reviews`` probe more often, not less. Does not touch
    ``consecutive_probe_failures``: that counter belongs to the reviewer
    quota's own probe path, not the flat-interval probe.

    Returns a new state dict; does not mutate ``data``.
    """
    if not is_reviewer_quota_exhausted(data):
        return data
    quota = _reviewer_quota(data)
    current = quota.get("probe_after")
    if current:
        try:
            current_dt = datetime.fromisoformat(current.replace("Z", "+00:00"))
            new_dt = datetime.fromisoformat(probe_after.replace("Z", "+00:00"))
            if current_dt >= new_dt:
                return data
        except (ValueError, TypeError):
            # Malformed current value: overwrite with the well-formed new one.
            pass
    quota["probe_after"] = probe_after
    return {**data, "reviewer_quota": quota}


def _quota_probe(data: dict[str, Any]) -> dict[str, Any]:
    """Return the quota-probe scheduling sub-dict from ``data``.

    Distinct from ``reviewer_quota``: this tracks the flat-interval probe
    mechanism itself (claude_code.run_quota_probe's schedule), not either
    throttle it checks up on. Ensures a mutable copy so callers can build new
    state without mutating the original ``data``.
    """
    probe = data.get("quota_probe")
    if not isinstance(probe, dict):
        return {}
    return dict(probe)


def is_quota_probe_armed(data: dict[str, Any]) -> bool:
    """True once a next-probe timestamp has been scheduled.

    Callers arm the probe (``arm_quota_probe``) the first pass a throttle
    indicator is observed, without probing that same pass, so the first
    real probe attempt happens roughly ``interval_minutes`` after onset
    rather than instantly.
    """
    return bool(_quota_probe(data).get("next_probe_at"))


def is_quota_probe_due(data: dict[str, Any]) -> bool:
    """True when an armed probe's wait interval has elapsed.

    False (not True) when unarmed -- an absent schedule means "not armed
    yet", which callers handle by arming rather than probing immediately;
    see ``is_quota_probe_armed``. Malformed timestamps are treated as due,
    so a corrupt value cannot wedge probing off forever.
    """
    next_at = _quota_probe(data).get("next_probe_at")
    if not next_at:
        return False
    try:
        next_time = datetime.fromisoformat(next_at.replace("Z", "+00:00"))
        return datetime.now(UTC) >= next_time
    except (ValueError, TypeError):
        return True


def arm_quota_probe(data: dict[str, Any], next_probe_at: str) -> dict[str, Any]:
    """Schedule the next quota-probe attempt.

    Returns a new state dict; does not mutate ``data``.
    """
    probe = _quota_probe(data)
    probe["next_probe_at"] = next_probe_at
    return {**data, "quota_probe": probe}


def disarm_quota_probe(data: dict[str, Any]) -> dict[str, Any]:
    """Clear the scheduled next-probe timestamp.

    Called both when a probe succeeds (so the next exhaustion episode arms
    fresh instead of reusing a stale timestamp) and when no throttle
    indicator is active any more (the cooldown expired naturally before a
    probe fired). Returns a new state dict; does not mutate ``data``.
    """
    probe = _quota_probe(data)
    if not probe:
        return data
    probe.pop("next_probe_at", None)
    return {**data, "quota_probe": probe}


def _worktree_reclamation(data: dict[str, Any]) -> dict[str, Any]:
    """Return the worktree-reclamation scheduling sub-dict from ``data``.

    Tracks the ``next_run_at`` timestamp that gates the cadence-gated
    ``clean_worktrees`` sweep fired from the fleet pass (issue #636). Returns a
    mutable copy so callers can build new state without mutating ``data``.
    """
    sched = data.get("worktree_reclamation")
    if not isinstance(sched, dict):
        return {}
    return dict(sched)


def is_worktree_reclamation_due(data: dict[str, Any]) -> bool:
    """True when the reclamation sweep's interval has elapsed.

    An absent schedule means "never run yet", which is treated as due so the
    first fleet pass after startup clears the existing backlog of merged-PR
    worktrees (the exact accumulation issue #636 exists to fix). Malformed
    timestamps are also treated as due so a corrupt value cannot wedge
    reclamation off forever.
    """
    next_at = _worktree_reclamation(data).get("next_run_at")
    if not next_at:
        return True
    try:
        next_time = datetime.fromisoformat(next_at.replace("Z", "+00:00"))
        return datetime.now(UTC) >= next_time
    except (ValueError, TypeError):
        return True


def schedule_worktree_reclamation(data: dict[str, Any], next_run_at: str) -> dict[str, Any]:
    """Set the next reclamation sweep timestamp.

    Called after a sweep runs (or is skipped as not-due-armed) so the next
    sweep fires roughly ``interval_minutes`` later rather than on the very next
    pass. Returns a new state dict; does not mutate ``data``.
    """
    sched = _worktree_reclamation(data)
    sched["next_run_at"] = next_run_at
    return {**data, "worktree_reclamation": sched}


def _reconcile_pass(data: dict[str, Any]) -> dict[str, Any]:
    """Return the periodic in-loop reconcile scheduling sub-dict from ``data``.

    Distinct from ``quota_probe``: tracks the merge-lane-recovery §6-B
    cadence (``OrchestratorApp._maybe_reconcile_drift``), not the quota
    probe. Ensures a mutable copy so callers can build new state without
    mutating the original ``data``.
    """
    section = data.get("reconcile_pass")
    if not isinstance(section, dict):
        return {}
    return dict(section)


def is_reconcile_due(data: dict[str, Any]) -> bool:
    """True when the periodic in-loop reconcile pass should run.

    Unlike the quota probe (which only arms once a throttle indicator is
    observed, deliberately delaying the first real probe), reconcile has no
    "is something wrong" precondition to wait on -- it is a plain periodic
    cadence, so an absent schedule (never run before, e.g. right after a
    fresh deploy) is treated as due immediately rather than requiring one
    full interval to elapse first. This matters for G1: the divergence class
    this closes has already been sitting unrepaired indefinitely, so the
    first pass after this lands should not wait `interval_minutes` before
    doing anything. Malformed timestamps are treated as due, mirroring
    ``is_quota_probe_due``, so a corrupt value cannot wedge reconcile off
    forever.
    """
    next_at = _reconcile_pass(data).get("next_reconcile_at")
    if not next_at:
        return True
    try:
        next_time = datetime.fromisoformat(next_at.replace("Z", "+00:00"))
        return datetime.now(UTC) >= next_time
    except (ValueError, TypeError):
        return True


def arm_reconcile_pass(data: dict[str, Any], next_reconcile_at: str) -> dict[str, Any]:
    """Schedule the next periodic in-loop reconcile attempt.

    Returns a new state dict; does not mutate ``data``.
    """
    section = _reconcile_pass(data)
    section["next_reconcile_at"] = next_reconcile_at
    return {**data, "reconcile_pass": section}


def _deescalation_pass(data: dict[str, Any]) -> dict[str, Any]:
    """Return the periodic de-escalation sweep scheduling sub-dict from ``data``.

    Mirrors ``_reconcile_pass``: tracks issue #783's
    ``OrchestratorApp._maybe_deescalate_mechanical`` cadence. Ensures a
    mutable copy so callers can build new state without mutating ``data``.
    """
    section = data.get("deescalation_pass")
    if not isinstance(section, dict):
        return {}
    return dict(section)


def is_deescalation_due(data: dict[str, Any]) -> bool:
    """True when the periodic de-escalation sweep should run.

    Same "absent schedule is due immediately" semantics as
    ``is_reconcile_due`` -- a fresh deploy should not wait a full interval
    before its first pass, and a malformed timestamp is treated as due
    rather than wedging the sweep off forever.
    """
    next_at = _deescalation_pass(data).get("next_deescalation_at")
    if not next_at:
        return True
    try:
        next_time = datetime.fromisoformat(next_at.replace("Z", "+00:00"))
        return datetime.now(UTC) >= next_time
    except (ValueError, TypeError):
        return True


def arm_deescalation_pass(data: dict[str, Any], next_deescalation_at: str) -> dict[str, Any]:
    """Schedule the next periodic de-escalation sweep attempt.

    Returns a new state dict; does not mutate ``data``.
    """
    section = _deescalation_pass(data)
    section["next_deescalation_at"] = next_deescalation_at
    return {**data, "deescalation_pass": section}


def is_operator_queue_review_due(data: dict[str, Any]) -> bool:
    """True when the operator-queue depth gauge should run (issue #1314 item 2).

    Same "absent schedule is due immediately" semantics as
    ``is_deescalation_due`` -- a fresh deploy or a 0-interval config (the
    default, meaning "every pass") is always due, and a malformed timestamp
    is treated as due rather than wedging the gauge off forever.
    """
    next_at = _deescalation_pass(data).get("next_operator_queue_review_at")
    if not next_at:
        return True
    try:
        next_time = datetime.fromisoformat(next_at.replace("Z", "+00:00"))
        return datetime.now(UTC) >= next_time
    except (ValueError, TypeError):
        return True


def arm_operator_queue_review(
    data: dict[str, Any], next_operator_queue_review_at: str
) -> dict[str, Any]:
    """Schedule the next operator-queue depth gauge check (issue #1314 item 2).

    Returns a new state dict; does not mutate ``data``.
    """
    section = _deescalation_pass(data)
    section["next_operator_queue_review_at"] = next_operator_queue_review_at
    return {**data, "deescalation_pass": section}


def any_quota_exhausted_indicator(data: dict[str, Any]) -> bool:
    """True when either throttle mechanism is currently active.

    The general/root throttle (``is_throttled``) and the reviewer-specific
    quota throttle (``is_reviewer_quota_exhausted``) are the two places a
    provider quota/rate-limit/auth exhaustion is recorded. General-purpose
    "is anything throttled" check; see ``is_quota_probe_actionable`` for the
    narrower gate the flat-interval probe itself uses.
    """
    return is_throttled(data) or is_reviewer_quota_exhausted(data)


def _root_throttle_is_claude_code_shaped(data: dict[str, Any]) -> bool:
    """True when the root throttle, if any, is one an ambient CLI probe speaks to.

    Excludes a ``provider_auth`` cooldown (a dead credential does not
    self-heal in minutes -- see claude_code._classify_session_failure) and
    excludes any adapter other than claude-code (devin uses a different tool
    entirely; the api adapter routes through a separately configured
    provider base_url/key, not the account an operator manually switches).
    ``None`` adapter_kind is included for backward compatibility with
    throttles set before ``throttle_adapter_kind`` existed.
    """
    return data.get("throttle_reason") != "provider_auth" and data.get(
        "throttle_adapter_kind"
    ) in (None, "claude-code")


def is_quota_probe_actionable(data: dict[str, Any]) -> bool:
    """True when a green ambient-CLI probe could actually clear something.

    Narrower than ``any_quota_exhausted_indicator``: mirrors exactly the
    condition ``clear_quota_throttles`` acts on, so arming/running the flat
    probe is impossible unless a green result would change state. Reviewer
    quota exhaustion always qualifies (always set from a Claude-family
    reviewer launch, never from the provider-auth pattern -- see
    dispatch_reviews); the root throttle qualifies only when it is
    claude-code-shaped. A devin/api-adapter or provider_auth-reasoned root
    throttle would survive a green probe untouched (see
    ``clear_quota_throttles``), so probing for one would just burn Haiku
    sessions every ``interval_minutes`` for the whole cooldown window
    without ever being able to help -- this is the gate that prevents that.
    """
    if is_reviewer_quota_exhausted(data):
        return True
    return is_throttled(data) and _root_throttle_is_claude_code_shaped(data)


def clear_quota_throttles(data: dict[str, Any]) -> dict[str, Any]:
    """Clear throttle state after a green quota probe.

    A green probe (a cheap ambient-CLI Haiku session that completed without
    hitting a throttle/auth signature) proves the Claude Code CLI's own
    ambient auth/quota has recovered -- e.g. the operator switched to a
    different subscription account. That is exactly the condition
    ``is_quota_probe_actionable`` gates on, which callers are expected to
    check before running a probe at all -- see that function's docstring
    for what is and is not cleared here and why.

    The ``last_probe_cleared_at`` recovery marker is only stamped when this
    call actually cleared something (a claude-code-shaped root throttle or
    reviewer-quota exhaustion), so a caller that bypasses the
    ``is_quota_probe_actionable`` guard cannot seed a false recovery marker.
    It is stored with microsecond precision so sub-second comparisons against
    a dead session's log mtime are not flipped (issue #662).

    Returns a new state dict; does not mutate ``data``.
    """
    cleared = data
    cleared_root = _root_throttle_is_claude_code_shaped(data)
    if cleared_root:
        cleared = {
            **cleared,
            "throttled_until": None,
            "throttle_reason": None,
            "throttle_adapter_kind": None,
        }
    cleared = clear_reviewer_quota(cleared)
    reviewer_quota = cleared.get("reviewer_quota") or {}
    # Stamp the recovery time so the dead-reviewer reap sweep can suppress
    # backoff for sessions whose throttle signature predates this recovery
    # (issue #662): a dead reviewer's log tail is frozen at death time and
    # does not reflect a quota window that has since reopened, so re-applying
    # backoff from it would re-poison reviewer_quota anchored to "now" rather
    # than the original death time. Only populate the marker when we actually
    # cleared a claude-code-shaped root throttle or reviewer_quota (the
    # exhaustion checks treat a missing throttled_until/probe_after as "not
    # exhausted", so an ever-present recovery marker is benign but only useful
    # when a real recovery happened).
    if is_reviewer_quota_exhausted(data) or cleared_root:
        reviewer_quota = {
            **reviewer_quota,
            "consecutive_probe_failures": 0,
            "last_probe_cleared_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        cleared = {**cleared, "reviewer_quota": reviewer_quota}
    return cleared
