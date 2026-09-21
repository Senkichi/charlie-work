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
* The cache key is the exact quadruple the four-condition predicate is a
  function of: ``pr_number`` (which PR), ``reviewed_head_sha`` (which
  approval), ``live_head_sha`` (which merged head), and the *effective*
  ``queue_bot_login`` (condition 4 of the predicate, and the config value
  that gates the whole check -- ``queue_sync_coverage._queue_sync_merge_
  covered`` returns ``covered=False`` outright when it is unset). Issue
  #1473 review finding 1: an earlier revision left this out of the key, so
  rotating or unsetting ``queue_bot_login`` (bot identity migration, or
  disabling recognition entirely because the bot account is suspected
  compromised) could not re-arm a merge already memoized under the old
  value -- the kill switch was a no-op for exactly the merges it exists to
  re-arm. Including it in the key is necessary but not sufficient by
  itself: :func:`charlie_work.orchestration.instrumentation_ops.
  OrchestratorApp._queue_sync_merge_covered` additionally skips the cache
  lookup outright when ``queue_bot_login`` is falsy, so the kill switch is
  enforced at the earliest point rather than only by key mismatch. A new
  push to a PR's branch, a rewritten review decision, a brand-new merged PR
  number, or a changed bot login all produce a key never seen before, so
  the cache cannot suppress detection of any of those -- it can only ever
  skip re-deriving an answer already reached for the exact same facts.
* A missing, unreadable, or structurally-corrupt cache file degrades to an
  EMPTY cache (fail closed): every triple reads as a miss, so the caller
  re-verifies from ``gh`` exactly as if memoization did not exist. Corruption
  can only cost extra API calls, never suppress a finding or fabricate a
  ``covered=True`` verdict.
* The write path (:func:`record_covered`) is best-effort in the same
  fail-safe direction: ``StateLockBusy`` *and* ``OSError`` (disk full, a
  read-only state dir, or -- on Windows -- a concurrent reader holding the
  destination open across ``Path.replace()``) are both caught, logged, and
  turned into a ``False`` return, never raised. Issue #1473 review finding
  2: catching only ``StateLockBusy`` let a transient write failure escape
  this function, and from there ``_detect_unauthorized_merges`` (which
  guards ``GitHubError`` only) and ``reap_loop.py`` (which guards nothing at
  that call site) -- so a disk-full condition could abort the whole reap
  pass instead of merely costing one avoidable re-check next pass.
* The caller is responsible for never consulting the cache at all under
  ``--dry-run`` (issue #1473 review finding 5): a dry-run pass must have the
  same events.db/state.json footprint as a caller that never ran, and a raw
  file write here bypasses ``WriteGate`` entirely, so this module cannot
  enforce that on its own.

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

Batch preload: :func:`load_cache_map` loads the whole file once so a caller
iterating several candidate PRs in one pass (``_detect_unauthorized_merges``)
can pass the same in-memory map to every :func:`is_covered_cached` call via
its ``preloaded`` parameter, instead of re-``open``-ing and re-``json.load``-
ing the file once per candidate (issue #1473 review finding 4 -- the file
is never pruned, so a per-candidate re-parse re-introduces a smaller version
of the same "unbounded per-pass work for an unchanging answer" shape the
issue is about).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .state import StateLockBusy, advisory_file_lock, utc_now

logger = logging.getLogger(__name__)


def _cache_key(
    pr_number: int, reviewed_head_sha: str, live_head_sha: str, queue_bot_login: str
) -> str:
    """Build the memoization key for one coverage verdict.

    All four components are load-bearing: dropping ``live_head_sha`` would
    let a later force-push to a merged PR's (undeleted) branch keep reading
    the old verdict; dropping ``reviewed_head_sha`` would let a corrected or
    re-written review decision keep reading a verdict computed against a
    different approval; dropping ``queue_bot_login`` (issue #1473 review
    finding 1) would let a verdict memoized under one bot identity keep
    reading as covered after the operator rotates or unsets it -- see the
    module docstring's SAFETY section.
    """
    return f"{pr_number}:{reviewed_head_sha}:{live_head_sha}:{queue_bot_login}"


def load_cache_map(path: Path) -> dict[str, str]:
    """Load the durable covered-verdict cache as a plain ``{key: timestamp}`` map.

    Returns ``{}`` (every triple a miss) on a missing, unreadable, or
    structurally wrong-shaped file. This is the fail-closed path described in
    the module docstring: a corrupt cache degrades to "verify everything
    again", never to "assume it is covered".

    Public (unlike the rest of this module's read/write helpers) so a caller
    that will look up several keys in one pass -- ``_detect_unauthorized_
    merges`` -- can load the file once and pass the result to every
    :func:`is_covered_cached` call via its ``preloaded`` parameter, rather
    than re-parsing the file once per candidate PR (issue #1473 review
    finding 4).
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
    queue_bot_login: str,
    preloaded: dict[str, str] | None = None,
) -> bool:
    """Return ``True`` iff this exact quadruple was already determined covered.

    Lock-free read: the on-disk file is only ever replaced atomically (see
    :func:`_save_cache`), so a concurrent writer can never be observed
    mid-write. There is no code path that writes a ``False`` entry, so a
    ``True`` here is proof a prior pass ran the full four-condition check and
    it fully passed under this exact ``queue_bot_login``.

    ``preloaded``, when given, is used in place of re-reading ``path`` (issue
    #1473 review finding 4) -- pass the result of one :func:`load_cache_map`
    call shared across every candidate PR in a single
    ``_detect_unauthorized_merges`` pass. ``None`` (the default) falls back
    to loading the file fresh, preserving this function's original
    single-call contract for any other caller.
    """
    key = _cache_key(pr_number, reviewed_head_sha, live_head_sha, queue_bot_login)
    entries = preloaded if preloaded is not None else load_cache_map(path)
    return key in entries


def record_covered(
    path: Path,
    *,
    pr_number: int,
    reviewed_head_sha: str,
    live_head_sha: str,
    queue_bot_login: str,
) -> bool:
    """Durably record a freshly-determined ``covered=True`` verdict.

    Locked read-modify-write (mirrors
    ``api_budget.settle_session_to_disk``): concurrent passes recording
    different PRs' verdicts cannot lose one another's entry. Idempotent --
    recording an already-present key is a no-op.

    Best-effort in the fail-safe direction: if the advisory lock is busy, or
    any step of the write (``mkdir``, opening the temp file, ``json.dump``,
    or the atomic ``replace()``) raises ``OSError`` -- disk full, a
    read-only state dir, or on Windows a concurrent reader holding the
    destination open across ``replace()`` -- the entry is simply not
    persisted this pass (logged, ``False`` returned, never raised). Issue
    #1473 review finding 2: this used to catch only ``StateLockBusy``, so an
    ``OSError`` from the write path escaped uncaught into
    ``_detect_unauthorized_merges`` (which guards ``GitHubError`` only) and
    from there into callers with no guard at all -- turning a memoization
    optimization into an aborted reap pass. This pass's caller has already
    received its ``covered=True`` answer from the real check before calling
    this function, so a skipped memoization only costs one avoidable
    re-check on the *next* pass -- it never changes any pass's
    ``unauthorized_merge_detected`` verdict.
    """
    key = _cache_key(pr_number, reviewed_head_sha, live_head_sha, queue_bot_login)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with advisory_file_lock(path):
            entries = load_cache_map(path)
            if key in entries:
                return True
            entries = {**entries, key: utc_now()}
            _save_cache(path, entries)
    except (StateLockBusy, OSError) as exc:
        logger.warning(
            "queue-sync coverage cache write failed at %s; skipping "
            "memoization of pr=%s (%s) (this pass's verdict is unaffected -- "
            "the check simply repeats next pass)",
            path,
            pr_number,
            exc,
        )
        return False
    return True


__all__ = ["is_covered_cached", "load_cache_map", "record_covered"]
