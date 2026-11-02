"""Bounded retry of a dead worker's failed salvage push (issue #2847).

The orphan sweep publishes a dead worker's committed-but-unpushed work once
(``worktree.salvage_push_stranded_commits``). A transient failure of that one
push -- ``push_failed: command timed out after 60s`` through a repo's pre-push
gate on a loaded host -- used to be final: nothing retried it, the next
redispatch found the commit sitting in the worktree, and the refuse-to-reset
guard classified it as divergent local work
(``worktree_unsafe_local_commits``, a judgment escalation, #807).

``create_worktree`` calls :func:`retry_failed_salvage_push` before any of its
guards. The retry runs only when ALL of these hold, so a real second writer
still escalates exactly as before:

- the newest salvage outcome recorded for the issue's branch is a
  ``salvage_push_failed`` whose error is a *timeout* of the push
  (:func:`is_transient_push_failure`);
- the issue has attempts left (``dispatch.salvage_push_retries``, counted in
  ``state["issues"][n]["salvage_push_retries"]``);
- ``salvage_push_stranded_commits`` still finds a push to make. It re-probes
  the remote every time and only fast-forwards (or creates an absent branch),
  so a remote that moved since the failure surfaces as a ``skip_reason``
  (``diverged``, ``remote_head_not_local``) rather than a push.

Transient means a timeout and nothing else, deliberately. ``push_branch``
records ``result.error or result.stderr`` and ``run_captured`` always sets
``error`` on a failure -- ``command exited N`` or ``command timed out after
Ns`` -- so a rejection's stderr (non-fast-forward, protected branch, a
failing pre-push or pre-receive hook) is never in the recorded text and
cannot be told apart from any other non-zero exit. Matching rejection wording
would never match; allow-listing the one signature that IS distinguishable
makes every other failure fall closed to the guards.

The retry's own result is then handled by cause (:data:`TRANSIENT_SKIPS`,
:data:`ON_ORIGIN_SKIPS`, :data:`FALL_THROUGH_SKIPS`): a skip that says the probe
itself failed defers the launch like a failed push; a skip that says the work
is already on origin counts as landed; a skip that says someone else writes
the branch falls through to the guards.

Kept out of ``worktree.py`` (file-size ratchet, #1442): the policy has no git
dependency of its own -- the caller passes the push as a callable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import state as _state

logger = logging.getLogger(__name__)

SALVAGE_PUSH_FAILED = "salvage_push_failed"
SALVAGE_PUSHED = "salvage_pushed_stranded_commits"
# ``SalvagePushResult.error`` is ``f"push_failed: {push_error}"``; a failure
# of any other shape (a skip reason, a probe failure) is not a failed push.
PUSH_FAILED_PREFIX = "push_failed:"
RETRY_COUNT_FIELD = "salvage_push_retries"

# ``run_captured``'s wording for a command that hit its timeout:
# ``command timed out after {N}s`` -- the one failure of the push that says
# nothing was refused. Every other recorded failure is ``command exited N``.
_TIMEOUT_MARKER = "timed out after"

# ``SalvagePushResult.skip_reason`` values, classified by what a retry should do
# about them. Every reason ``salvage_push_stranded_commits`` can return is in
# exactly one set (``tests/test_salvage_retry.py`` derives the list from its
# source), and a reason in none of them falls through like ``FALL_THROUGH_SKIPS``.
#
# The probe itself failed: nothing was learned about the remote, so the work may
# still be stranded and a later attempt can succeed. Defer while attempts remain.
TRANSIENT_SKIPS = frozenset({"remote_probe_failed", "count_failed", "base_unresolvable"})
# The remote already holds the worker's tip (an earlier push landed even though
# its verification timed out): nothing is stranded any more.
ON_ORIGIN_SKIPS = frozenset({"up_to_date", "no_stranded_commits"})
# Someone else may be writing the branch, or there is nothing to publish: leave
# the decision to the refuse-to-reset guards, exactly as before this retry.
FALL_THROUGH_SKIPS = frozenset(
    {
        "diverged",
        "remote_head_not_local",
        "live_writer_marker",
        "operator_claimed",
        "branch_ref_missing",
        "no_worktree",
        "invalid_branch",
        "no_commits_beyond_base",
    }
)


def skip_kind(skip_reason: str | None) -> str:
    """The reason name without detail (``invalid_branch: <exc>`` -> ``invalid_branch``)."""
    return (skip_reason or "").partition(":")[0]


def is_transient_push_failure(error: str | None) -> bool:
    """True when ``error`` is a timeout of the salvage push itself.

    An allow-list on purpose: see the module docstring for why a rejection
    cannot be recognised in the text production records, so anything that is
    not positively a timeout is treated as final.
    """
    text = error or ""
    return text.startswith(PUSH_FAILED_PREFIX) and _TIMEOUT_MARKER in text


def landed_on_origin(result: Any) -> bool:
    """True when ``result`` (a ``SalvagePushResult``) shows the work on origin."""
    return bool(result.pushed) or skip_kind(result.skip_reason) in ON_ORIGIN_SKIPS


@dataclass(frozen=True)
class SalvageRetryOutcome:
    """What one :func:`retry_failed_salvage_push` call did.

    ``attempted`` is False when no push was tried (nothing recorded, budget
    spent, a non-transient failure on record, kill switch off, bookkeeping
    unavailable); the caller's guards then decide exactly as they did before
    this retry existed. ``error`` / ``skip_reason`` carry the push result when
    it was attempted.
    """

    attempted: bool = False
    pushed: bool = False
    attempt: int = 0
    limit: int = 0
    error: str | None = None
    skip_reason: str | None = None

    @property
    def on_origin(self) -> bool:
        """The worker's commits are on origin: pushed now, or already there."""
        return self.attempted and (self.pushed or skip_kind(self.skip_reason) in ON_ORIGIN_SKIPS)

    @property
    def defer(self) -> bool:
        """True when the launch is worth deferring for another redispatch.

        The push timed out, or the probe could not tell, and attempts remain.
        A refused push, a moved remote or a live writer never defers.
        """
        if not self.attempted or self.on_origin or self.attempt >= self.limit:
            return False
        return (
            is_transient_push_failure(self.error) or skip_kind(self.skip_reason) in TRANSIENT_SKIPS
        )

    @property
    def reason(self) -> str | None:
        """Why the attempt did not land the work, for messages."""
        return self.error or self.skip_reason


def _event_matches(event: Any, issue_number: int, branch: str) -> Mapping[str, Any] | None:
    if not isinstance(event, dict):
        return None
    payload = event.get("payload")
    if (
        not isinstance(payload, dict)
        or payload.get("issue_number") != issue_number
        or payload.get("branch") != branch
    ):
        return None
    return payload


def latest_retryable_failure(
    state: Mapping[str, Any], issue_number: int, branch: str
) -> Mapping[str, Any] | None:
    """Payload of the newest salvage outcome for this branch, if it is a retryable failure.

    Derived from the event ring, so there is no marker to go stale: a later
    ``salvage_pushed_stranded_commits`` for the branch (the work is on origin)
    or a non-transient failure supersedes an older failure. ``None`` when the
    newest outcome is anything but a ``push_failed`` that timed out.
    """
    for event in reversed(state.get("events") or []):
        payload = _event_matches(event, issue_number, branch)
        if payload is None:
            continue
        kind = event.get("kind")
        if kind == SALVAGE_PUSHED:
            return None
        if kind == SALVAGE_PUSH_FAILED:
            error = payload.get("error")
            return payload if isinstance(error, str) and is_transient_push_failure(error) else None
    return None


def _attempts_used(state: Mapping[str, Any], issue_number: int) -> int:
    entry = (state.get("issues") or {}).get(str(issue_number))
    used = entry.get(RETRY_COUNT_FIELD) if isinstance(entry, dict) else 0
    return used if isinstance(used, int) and not isinstance(used, bool) and used > 0 else 0


def _with_issue_fields(state: dict[str, Any], issue_number: int, **fields: Any) -> dict[str, Any]:
    """A copy of ``state`` with ``fields`` set on the issue entry; ``None`` drops a field."""
    issues = dict(state.get("issues") or {})
    entry = dict(issues.get(str(issue_number)) or {})
    for name, value in fields.items():
        if value is None:
            entry.pop(name, None)
        else:
            entry[name] = value
    issues[str(issue_number)] = entry
    return {**state, "issues": issues}


def _claim_attempt(state_file: Path, issue_number: int, branch: str, limit: int) -> int | None:
    """Spend one attempt before the push starts; ``None`` when no retry is owed.

    Counted up front, under the state lock, so an orchestrator that dies
    mid-push cannot retry forever without the count ever moving.
    """
    with _state.state_lock(state_file):
        state = _state.load_state(state_file)
        if latest_retryable_failure(state, issue_number, branch) is None:
            return None
        attempt = _attempts_used(state, issue_number) + 1
        if attempt > limit:
            return None
        # write-gate-exempt(issue=2847): no write_gate receiver in create_worktree (config + state_file only); the preflight runs only on a real launch, which also creates the worktree
        _state.save_state(
            state_file, _with_issue_fields(state, issue_number, **{RETRY_COUNT_FIELD: attempt})
        )
        return attempt


def _record_outcome(
    state_file: Path,
    issue_number: int,
    branch: str,
    attempt: int,
    result: Any,
) -> None:
    """Append the retry's outcome as the same event kinds the sweep writes.

    Reusing ``salvage_push_failed`` / ``salvage_pushed_stranded_commits`` keeps
    :func:`latest_retryable_failure` reading one newest outcome, whoever wrote
    it, and the dashboard rollup unchanged. ``retry_attempt`` marks these.

    A skip that finds the work already on origin (an earlier push landed and
    only its verification timed out) is recorded as the landed outcome with
    ``already_on_origin`` set, so the stale failure stops being the newest
    outcome and later launches do not spend attempts re-discovering it.
    """
    payload: dict[str, Any] = {
        "issue_number": issue_number,
        "pr_number": None,
        "branch": branch,
        "old_remote_sha": result.old_remote_sha,
        "commit_count": result.commit_count,
        "retry_attempt": attempt,
    }
    if landed_on_origin(result):
        kind = SALVAGE_PUSHED
        if result.pushed:
            payload["new_remote_sha"] = result.new_remote_sha
        else:
            payload["new_remote_sha"] = result.old_remote_sha
            payload["already_on_origin"] = True
    elif result.error:
        kind = SALVAGE_PUSH_FAILED
        payload["error"] = result.error
    else:
        # A skip that is not a landing (probe failure, moved remote, live
        # writer, ...): no push was made and nothing failed, so there is no
        # salvage outcome to record. The claimed attempt stays spent and the
        # older failure stays the newest outcome; the caller decides by cause.
        return
    with _state.state_lock(state_file):
        state = _state.load_state(state_file)
        # Work on origin ends the episode: drop the count so a later,
        # unrelated stranding gets a fresh budget.
        if kind == SALVAGE_PUSHED:
            state = _with_issue_fields(state, issue_number, **{RETRY_COUNT_FIELD: None})
        # write-gate-exempt(issue=2847): no write_gate receiver in create_worktree (config + state_file only); the preflight runs only on a real launch, which also creates the worktree
        state = _state.append_event(state, kind, payload, state_path=state_file)
        # write-gate-exempt(issue=2847): same preflight bookkeeping as the append_event above
        _state.save_state(state_file, state)


def retry_failed_salvage_push(
    state_file: Path,
    issue_number: int,
    branch: str,
    *,
    limit: int,
    push: Callable[[], Any],
) -> SalvageRetryOutcome:
    """Retry the salvage push once if a transient failure of it is on record.

    ``push`` is a zero-argument callable returning a
    ``worktree.SalvagePushResult`` (``salvage_push_stranded_commits`` bound to
    the worktree); it must not raise. ``limit`` is
    ``dispatch.salvage_push_retries`` -- 0 turns the retry off.

    Bookkeeping that cannot be read or written never raises out of here: an
    unreadable claim means no retry (the caller's guards then behave exactly
    as they did before this existed), and a failed outcome write after the
    push has run still reports that push.
    """
    if limit <= 0:
        return SalvageRetryOutcome()
    try:
        attempt = _claim_attempt(state_file, issue_number, branch, limit)
    except Exception:  # noqa: BLE001 -- bookkeeping must never block a launch
        logger.warning(
            "salvage retry bookkeeping failed for issue #%s; not retrying",
            issue_number,
            exc_info=True,
        )
        return SalvageRetryOutcome()
    if attempt is None:
        return SalvageRetryOutcome()

    result = push()
    try:
        _record_outcome(state_file, issue_number, branch, attempt, result)
    except Exception:  # noqa: BLE001 -- the push already happened; do not lose it
        logger.warning(
            "could not record salvage retry %d/%d for issue #%s",
            attempt,
            limit,
            issue_number,
            exc_info=True,
        )
    return SalvageRetryOutcome(
        attempted=True,
        pushed=bool(result.pushed),
        attempt=attempt,
        limit=limit,
        error=result.error,
        skip_reason=result.skip_reason,
    )
