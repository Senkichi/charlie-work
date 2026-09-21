"""Durable memoization for ``_queue_sync_merge_covered`` verdicts (issue #1473).

``_queue_sync_merge_covered`` (``queue_sync_coverage.py``) answers a question
about a *merged, immutable* commit: "is this merge head a queue-sync of the
approved head?" Once determined ``covered=True`` for a given
``(pr_number, reviewed_head_sha, live_head_sha)`` triple, the answer can never
change -- the underlying git objects are content-addressed and the PR is
closed. Without memoization, ``_detect_unauthorized_merges`` re-pays the
three ``gh`` API calls (two ``commit()``, one ``compare()``) behind that
predicate, and re-emits the ``unauthorized_merge_queue_sync_covered`` audit
event, every loop pass for as long as the PR stays inside
``merged_pr_list()``'s 500-PR window -- measured at 5,723-12,208
``unauthorized_merge_queue_sync_covered`` rows/24h in a fleet repo, all
re-verifying the same handful of already-known-covered merges.

SAFETY (this is #502-tripwire-adjacent, security-relevant code):

* Only a fully-determined ``covered=True`` verdict is ever written here.
  ``covered=False`` (whether determined-not-covered or indeterminate/fetch-
  failed) is NEVER cached -- an uncovered or ambiguous merge must be
  re-examined every single pass, exactly as before this module existed.
* The cache key is the exact triple the four-condition predicate is a
  function of: ``pr_number`` (which PR), ``reviewed_head_sha`` (which
  approval), ``live_head_sha`` (which merged head). A new push to a PR's
  branch, a rewritten review decision, or a brand-new merged PR number all
  produce a key never seen before, so the cache cannot suppress detection of
  any of those -- it can only ever skip re-deriving an answer already reached
  for the exact same three facts.
* A missing, unreadable, or structurally-corrupt cache file degrades to an
  EMPTY cache (fail closed): every triple reads as a miss, so the caller
  re-verifies from ``gh`` exactly as if memoization did not exist. Corruption
  can only cost extra API calls, never suppress a finding or fabricate a
  ``covered=True`` verdict.

Persistence follows this repo's existing sidecar-cache shape
(``api_budget.settle_session_to_disk`` / ``fleet_health_baseline``): a small
JSON file next to ``state.json``, written via temp-file + ``replace()``
(CLAUDE.md's atomic-write invariant), with the read-modify-write on the
*write* path serialized by ``state.advisory_file_lock`` so two passes can
never lose one another's memoized entry. Reads are lock-free -- an atomic
``replace()`` never exposes a torn file to a concurrent reader, and
``state.load_state_locked`` vs. plain ``state.load_state`` draws the same
distinction for ``state.json`` itself.

Known trade-off, stated plainly: entries are never garbage-collected here.
A cache entry for a PR that later ages out of ``merged_pr_list()``'s 500-PR
window becomes dead weight (never looked up again), but the file stays a
small flat map of short strings -- even several thousand entries is a
trivially small JSON file, and pruning it would need to duplicate
``merged_pr_list()``'s own windowing policy for no operational benefit.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .state import StateLockBusy, advisory_file_lock, utc_now

logger = logging.getLogger(__name__)


def _cache_key(pr_number: int, reviewed_head_sha: str, live_head_sha: str) -> str:
    """Build the memoization key for one coverage verdict.

    All three components are load-bearing: dropping ``live_head_sha`` would
    let a later force-push to a merged PR's (undeleted) branch keep reading
    the old verdict; dropping ``reviewed_head_sha`` would let a corrected or
    re-written review decision keep reading a verdict computed against a
    different approval.
    """
    return f"{pr_number}:{reviewed_head_sha}:{live_head_sha}"


def _load_cache(path: Path) -> dict[str, str]:
    """Load the durable covered-verdict cache.

    Returns ``{}`` (every triple a miss) on a missing, unreadable, or
    structurally wrong-shaped file. This is the fail-closed path described in
    the module docstring: a corrupt cache degrades to "verify everything
    again", never to "assume it is covered".
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, LookupError, ValueError):
        logger.warning(
            "queue-sync coverage cache %s unreadable; treating as empty "
            "(every previously-covered PR re-verifies this pass)",
            path,
        )
        return {}
    entries = data.get("covered") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        return {}
    return {str(k): str(v) for k, v in entries.items() if isinstance(v, str)}


def _save_cache(path: Path, entries: dict[str, str]) -> None:
    """Atomically persist the cache (temp-file + ``replace()``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "generated_at": utc_now(), "covered": entries}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def is_covered_cached(
    path: Path,
    *,
    pr_number: int,
    reviewed_head_sha: str,
    live_head_sha: str,
) -> bool:
    """Return ``True`` iff this exact triple was already determined covered.

    Lock-free read: the on-disk file is only ever replaced atomically (see
    :func:`_save_cache`), so a concurrent writer can never be observed
    mid-write. There is no code path that writes a ``False`` entry, so a
    ``True`` here is proof a prior pass ran the full four-condition check and
    it fully passed.
    """
    key = _cache_key(pr_number, reviewed_head_sha, live_head_sha)
    return key in _load_cache(path)


def record_covered(
    path: Path,
    *,
    pr_number: int,
    reviewed_head_sha: str,
    live_head_sha: str,
) -> bool:
    """Durably record a freshly-determined ``covered=True`` verdict.

    Locked read-modify-write (mirrors
    ``api_budget.settle_session_to_disk``): concurrent passes recording
    different PRs' verdicts cannot lose one another's entry. Idempotent --
    recording an already-present key is a no-op.

    Best-effort: if the advisory lock is busy, the entry is simply not
    persisted this pass (logged, ``False`` returned, never raised). This
    pass's caller has already received its ``covered=True`` answer from the
    real check before calling this function, so a skipped memoization only
    costs one avoidable re-check on the *next* pass -- it never changes any
    pass's ``unauthorized_merge_detected`` verdict.
    """
    key = _cache_key(pr_number, reviewed_head_sha, live_head_sha)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with advisory_file_lock(path):
            entries = _load_cache(path)
            if key in entries:
                return True
            entries = {**entries, key: utc_now()}
            _save_cache(path, entries)
    except StateLockBusy:
        logger.warning(
            "queue-sync coverage cache lock busy at %s; skipping memoization "
            "of pr=%s (this pass's verdict is unaffected -- the check simply "
            "repeats next pass)",
            path,
            pr_number,
        )
        return False
    return True


__all__ = ["is_covered_cached", "record_covered"]
