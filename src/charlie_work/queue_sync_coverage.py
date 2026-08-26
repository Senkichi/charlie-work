"""Queue sync-merge coverage classification (issue #1194, retry fix PR).

Extracted from ``workflow.py`` (file-size ratchet, issue #1442): the
per-leg fetch-and-classify logic behind ``OrchestratorApp.
_queue_sync_merge_covered`` grew the monolith past its high-water mark, so
it moves here verbatim (only the ``self.gh`` / ``self.config`` / ``self.paths``
implicit access becomes explicit ``gh`` / ``queue_bot_login`` / ``state_file``
parameters, mirroring the ``_escalate_issue`` free-function precedent in
``escalation.py``). ``workflow.py`` re-exports every symbol here via a facade
import block, and ``OrchestratorApp._queue_sync_merge_covered`` /
``_fetch_commit_retrying`` / ``_fetch_compare_retrying`` stay as thin
delegating methods (the same shape already used for ``_write_rework_prompt``)
so existing call sites, tests, and any future monkeypatch of the bound method
keep working unchanged.

Issue #1194: Aviator's mergequeue syncs a PR branch with main before merging,
so the merged head is a bot-authored merge commit whose parents are the
approved head and a main commit -- a structural false positive for the #502
tripwire's strict SHA equality. ``_queue_sync_merge_covered`` recognizes that
shape, and only that shape, as covered by the recorded approval.

Retry fix: the three ``gh`` API calls this makes (two ``commit()`` lookups,
one ``compare()``) used to fail the whole four-condition check closed on ANY
failure of any one of them -- including a purely transient failure (rate
limit, TLS blip, timeout) that says nothing about whether the shape is
actually covered. Evidence: job-cannon PRs #1888, #1916, #1904, #1895 each
logged 326-376 ``unauthorized_merge_queue_sync_covered`` events against
exactly one ``unauthorized_merge_detected`` -- overwhelmingly a covered
shape, with one pass where some leg's ``gh api`` call failed transiently
(which leg is not recoverable from that evidence and is not claimed here).
Only FETCH failures are retried (commit() not ok, or compare() returning
None); a determined non-covered shape never retries -- retrying a query that
already answered would not change the answer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .github import GitHubRunResult
from .instrumentation import log_event

# Retry budget for the transient-fetch legs of _queue_sync_merge_covered
# (commit() x2, compare() x1). A single flaky `gh api` call on any leg used
# to fail the whole four-condition check closed indistinguishably from a
# genuinely-uncovered shape, misreporting a transient GitHub API blip as an
# unauthorized_merge_detected finding (evidence: job-cannon PRs #1888, #1916,
# #1904, #1895 each logged 326-376 unauthorized_merge_queue_sync_covered
# events against exactly one unauthorized_merge_detected -- overwhelmingly a
# covered shape, with one pass where some leg's gh api call failed
# transiently; which leg is not recoverable from that evidence). Only FETCH
# failures are retried (commit() not ok, or compare() returning None); a
# determined non-covered shape (wrong parent count, identity mismatch,
# "ahead"/"diverged" status, ...) never retries -- retrying a query that
# already answered would not change the answer.
_QUEUE_SYNC_RETRY_ATTEMPTS = 3
_QUEUE_SYNC_RETRY_BACKOFF_SECONDS = (1, 2)

# Patchable indirection so tests never sleep for real: tests replace this
# module-level name (monkeypatch.setattr(queue_sync_coverage,
# "_QUEUE_SYNC_RETRY_SLEEP", fake)) rather than mocking time.sleep globally.
_QUEUE_SYNC_RETRY_SLEEP = time.sleep


@dataclass(frozen=True)
class _QueueSyncCoverageResult:
    """Outcome of `_queue_sync_merge_covered`'s per-leg fetch-and-classify.

    `covered=True` iff all four conditions in `_queue_sync_merge_covered`'s
    docstring held after retries. When `covered=False`, `indeterminate`
    distinguishes *why* the caller should still fail closed:

    - `indeterminate=True`: a FETCH leg (commit() or compare()) never
      succeeded after retrying -- the shape could not be evaluated, not that
      it was evaluated and rejected. `reason` names the leg and carries the
      last error.
    - `indeterminate=False`: the shape was fully evaluated and determined not
      to be a covered queue sync-merge (e.g. wrong parent count, identity
      mismatch, comparison status "ahead"/"diverged"). `reason` names which
      condition failed.

    Both cases still fail closed (the caller still emits
    unauthorized_merge_detected) -- this dataclass only makes the two reasons
    distinguishable in the event payload instead of collapsing both into the
    same silent False.
    """

    covered: bool
    indeterminate: bool = False
    reason: str = ""


def _fetch_commit_retrying(gh: Any, sha: str, *, leg: str) -> tuple[dict[str, Any] | None, str]:
    """Fetch a commit via ``gh.commit()``, retrying only on fetch failure.

    Returns ``(commit_dict, "")`` on success, or ``(None, "<leg>: <error>")``
    if every attempt in ``_QUEUE_SYNC_RETRY_ATTEMPTS`` failed to fetch --
    a FETCH failure (rate limit, TLS blip, transport error), not a
    determined answer, so it is retried with
    ``_QUEUE_SYNC_RETRY_BACKOFF_SECONDS`` backoff between attempts.
    """
    last_error = ""
    for attempt in range(_QUEUE_SYNC_RETRY_ATTEMPTS):
        result = gh.commit(sha)
        if isinstance(result, GitHubRunResult) and result.ok and isinstance(result.value, dict):
            return result.value, ""
        last_error = (
            result.error
            if isinstance(result, GitHubRunResult) and result.error
            else f"commit {sha} fetch failed"
        )
        if attempt < _QUEUE_SYNC_RETRY_ATTEMPTS - 1:
            _QUEUE_SYNC_RETRY_SLEEP(_QUEUE_SYNC_RETRY_BACKOFF_SECONDS[attempt])
    return None, f"{leg}: {last_error}"


def _fetch_compare_retrying(
    gh: Any, base: str, head: str, *, leg: str
) -> tuple[dict[str, Any] | None, str]:
    """Fetch a comparison via ``gh.compare()``, retrying only on fetch failure.

    ``compare()`` returns ``None`` on any failure (errors as values, not
    raised) -- indistinguishable at that boundary from a real ``None``
    meaning "no data", so every ``None`` is treated as a FETCH failure
    and retried, same as ``_fetch_commit_retrying``.
    """
    last_error = ""
    for attempt in range(_QUEUE_SYNC_RETRY_ATTEMPTS):
        result = gh.compare(base, head)
        if isinstance(result, dict):
            return result, ""
        last_error = "compare returned no data"
        if attempt < _QUEUE_SYNC_RETRY_ATTEMPTS - 1:
            _QUEUE_SYNC_RETRY_SLEEP(_QUEUE_SYNC_RETRY_BACKOFF_SECONDS[attempt])
    return None, f"{leg}: {last_error}"


def _queue_sync_merge_covered(
    gh: Any,
    state_file: Path,
    queue_bot_login: str | None,
    pr: dict[str, Any],
    reviewed_head_sha: str | None,
    live_head_sha: str | None,
) -> _QueueSyncCoverageResult:
    """Classify whether the merged head is an approval-covered queue sync-merge.

    Issue #1194: Aviator's mergequeue syncs a PR branch with main before
    merging, so the merged head is a bot-authored merge commit whose
    parents are the approved head and a main commit — a structural false
    positive for the #502 tripwire's strict SHA equality. Recognize that
    shape, and only that shape, as covered by the recorded approval. All
    four conditions must hold (fail closed on every missing or ambiguous
    signal — an unanswerable question keeps the finding firing, and the
    existing ack flow remains the escape hatch):

    1. the live head is a merge commit with exactly two parents;
    2. exactly one parent IS the approved ``reviewed_head_sha``;
    3. the other parent is reachable from the base branch as it stood
       immediately BEFORE this PR's merge — anchored at the merge
       commit's first parent, NOT at current main. Post-merge, current
       main reaches everything the PR carried (including a smuggled
       second parent) through the merge commit itself, so a naive
       "reachable from main" test is vacuously true and enforces
       nothing. Reachability from pre-merge main is the discriminating
       form: nothing this PR introduced can be reachable from there.
       This condition is the load-bearing one — it bounds the merged
       content to (approved head + prior main) regardless of who
       authored the commit;
    4. the merge commit's author login is the configured
       ``auto_merge.queue_bot_login`` and its committer is GitHub's
       web-flow (both identity signals, same rationale as
       ``_verify_synced_head``: either alone is spoofable via crafted
       git metadata). Identity is defense-in-depth on top of (3), not a
       substitute for it. Unset ``queue_bot_login`` disables recognition
       entirely — the tripwire behaves exactly as before #1194.

    Suppressions are audit-logged (``unauthorized_merge_queue_sync_covered``,
    events.db via ``log_event`` — this path holds no state lock) so every
    exercised gate exception leaves a queryable trail; consumer is the
    operator auditing tripwire behavior, mirroring the skip-event pattern.

    Returns a ``_QueueSyncCoverageResult`` rather than a bare ``bool``:
    the three ``gh`` API calls this makes (two ``commit()``, one
    ``compare()``) each retry transient FETCH failures up to
    ``_QUEUE_SYNC_RETRY_ATTEMPTS`` times, but a determined non-covered
    shape never retries. If a fetch leg still fails after retrying, the
    result fails closed (``covered=False``) exactly as before, but is
    marked ``indeterminate=True`` so the caller's
    ``unauthorized_merge_detected`` payload can say "the API never
    answered" instead of misreporting "the shape was checked and
    rejected".
    """
    if not queue_bot_login:
        return _QueueSyncCoverageResult(
            covered=False, reason="auto_merge.queue_bot_login not configured"
        )
    if not reviewed_head_sha or not live_head_sha:
        return _QueueSyncCoverageResult(
            covered=False, reason="missing reviewed_head_sha or live_head_sha"
        )

    head_commit, head_error = _fetch_commit_retrying(gh, live_head_sha, leg="live_head_commit")
    if head_commit is None:
        return _QueueSyncCoverageResult(covered=False, indeterminate=True, reason=head_error)

    parents = [
        str(p.get("sha"))
        for p in (head_commit.get("parents") or [])
        if isinstance(p, dict) and p.get("sha")
    ]
    if len(parents) != 2:
        return _QueueSyncCoverageResult(
            covered=False, reason=f"live head has {len(parents)} parent(s), expected 2"
        )
    matching = [p for p in parents if p == reviewed_head_sha]
    if len(matching) != 1:
        # Zero matches: not a sync of the approved head. Two matches: a
        # degenerate both-parents-approved merge — nothing to sync, so
        # nothing this path needs to bless; fail closed.
        return _QueueSyncCoverageResult(
            covered=False,
            reason=(f"{len(matching)} of 2 parents match reviewed_head_sha, expected exactly 1"),
        )
    other_parent = next(p for p in parents if p != reviewed_head_sha)

    # Identity (condition 4). Checked before the extra API calls of
    # condition 3 purely to keep the miss path cheap; order does not
    # affect the verdict since all conditions are conjunctive.
    author = head_commit.get("author")
    author_login = author.get("login") if isinstance(author, dict) else None
    committer = head_commit.get("committer")
    committer_login = committer.get("login") if isinstance(committer, dict) else None
    commit_meta = head_commit.get("commit")
    commit_committer = commit_meta.get("committer") if isinstance(commit_meta, dict) else None
    committer_name = commit_committer.get("name") if isinstance(commit_committer, dict) else None
    if (
        author_login != queue_bot_login
        or committer_login != "web-flow"
        or committer_name != "GitHub"
    ):
        return _QueueSyncCoverageResult(
            covered=False,
            reason=(
                f"identity mismatch: author={author_login!r} "
                f"committer={committer_login!r} committer_name={committer_name!r}"
            ),
        )

    # Condition 3: anchor at pre-merge main via the landing merge
    # commit's first parent. GitHub commits merges on the base branch,
    # so parents[0] of merge_commit_sha is the base tip this merge
    # advanced. If the queue fast-forwarded instead (merge commit == the
    # sync commit itself), parents[0] is the approved head, the compare
    # below cannot succeed, and the finding keeps firing — fail closed.
    merge_commit_sha = pr.get("mergeCommitOid")
    if not merge_commit_sha:
        return _QueueSyncCoverageResult(covered=False, reason="pr missing mergeCommitOid")
    landing_commit, landing_error = _fetch_commit_retrying(
        gh, str(merge_commit_sha), leg="landing_commit"
    )
    if landing_commit is None:
        return _QueueSyncCoverageResult(covered=False, indeterminate=True, reason=landing_error)
    landing_parents = [
        str(p.get("sha"))
        for p in (landing_commit.get("parents") or [])
        if isinstance(p, dict) and p.get("sha")
    ]
    if not landing_parents:
        return _QueueSyncCoverageResult(covered=False, reason="landing commit has no parents")
    pre_merge_base = landing_parents[0]

    comparison, compare_error = _fetch_compare_retrying(
        gh, pre_merge_base, other_parent, leg="compare"
    )
    if comparison is None:
        return _QueueSyncCoverageResult(covered=False, indeterminate=True, reason=compare_error)
    # "identical"/"behind" mean other_parent introduces zero commits not
    # already on pre-merge main; "ahead"/"diverged" (or anything else)
    # mean it carries content this approval never covered.
    status = comparison.get("status")
    if status not in ("identical", "behind"):
        return _QueueSyncCoverageResult(
            covered=False,
            reason=f"compare status {status!r}, expected identical or behind",
        )

    log_event(
        state_file,
        "unauthorized_merge_queue_sync_covered",
        {
            "pr": pr.get("number"),
            "reviewed_head_sha": reviewed_head_sha,
            "live_head_sha": live_head_sha,
            "sync_parent": other_parent,
            "pre_merge_base": pre_merge_base,
            "queue_bot_login": queue_bot_login,
        },
    )
    return _QueueSyncCoverageResult(covered=True)
