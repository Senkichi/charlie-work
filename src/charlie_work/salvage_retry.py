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
  ``salvage_push_failed`` whose error is a ``push_failed`` and not a rejection;
- the issue has attempts left (``dispatch.salvage_push_retries``, counted in
  ``state["issues"][n]["salvage_push_retries"]``);
- ``salvage_push_stranded_commits`` still finds a push to make. It re-probes
  the remote every time and only fast-forwards (or creates an absent branch),
  so a remote that moved since the failure surfaces as a ``skip_reason``
  (``diverged``, ``remote_head_not_local``) rather than a push.

The rejection check on the recorded text is a cheap first filter, not the
safety property: ``push_branch`` records ``result.error or result.stderr`` and
``run_captured`` always sets ``error`` to ``command exited N`` on a non-zero
exit, so a real rejection's stderr is normally NOT in the recorded text. The
re-probe above is what stops a second writer.

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

# Git's own wording for a push the server refused (non-fast-forward, stale
# info, protected branch). Matched case-insensitively against the recorded
# error; see the module docstring for why this is only a first filter.
_REJECTION_MARKERS: tuple[str, ...] = (
    "rejected",
    "non-fast-forward",
    "fetch first",
    "protected branch",
)


@dataclass(frozen=True)
class SalvageRetryOutcome:
    """What one :func:`retry_failed_salvage_push` call did.

    ``attempted`` is False when no push was tried (nothing recorded, budget
    spent, rejection on record, kill switch off, bookkeeping unavailable);
    the caller's guards then decide exactly as they did before this retry
    existed. ``defer`` is True only for a failed push that is worth another
    redispatch: the error is not a rejection and attempts remain.
    """

    attempted: bool = False
    pushed: bool = False
    attempt: int = 0
    limit: int = 0
    error: str | None = None
    defer: bool = False


def is_rejection(error: str | None) -> bool:
    """True when ``error`` reads as the server refusing the push."""
    text = (error or "").lower()
    return any(marker in text for marker in _REJECTION_MARKERS)


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
    ``salvage_pushed_stranded_commits`` for the branch (the push landed) or a
    rejection supersedes an older failure. ``None`` when the newest outcome is
    anything but a non-rejected ``push_failed``.
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
            if (
                isinstance(error, str)
                and error.startswith(PUSH_FAILED_PREFIX)
                and not is_rejection(error)
            ):
                return payload
            return None
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
    """
    payload: dict[str, Any] = {
        "issue_number": issue_number,
        "pr_number": None,
        "branch": branch,
        "old_remote_sha": result.old_remote_sha,
        "commit_count": result.commit_count,
        "retry_attempt": attempt,
    }
    if result.pushed:
        kind = SALVAGE_PUSHED
        payload["new_remote_sha"] = result.new_remote_sha
    elif result.error:
        kind = SALVAGE_PUSH_FAILED
        payload["error"] = result.error
    else:
        # A skip (diverged remote, live writer, ...): no push was made and
        # nothing failed, so there is no salvage outcome to record. The
        # claimed attempt stays spent; the caller's guards decide.
        return
    with _state.state_lock(state_file):
        state = _state.load_state(state_file)
        # A landed push ends the episode: drop the count so a later,
        # unrelated stranding gets a fresh budget.
        if result.pushed:
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

    Never raises: bookkeeping that cannot be read or written means no retry,
    and the caller's guards then behave exactly as they did before this
    existed.
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
    error = result.error
    return SalvageRetryOutcome(
        attempted=True,
        pushed=bool(result.pushed),
        attempt=attempt,
        limit=limit,
        error=error,
        defer=bool(error) and not is_rejection(error) and attempt < limit,
    )
