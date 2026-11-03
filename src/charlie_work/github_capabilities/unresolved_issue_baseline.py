"""Durable once-only baseline for ``github_issue_state_partial_fallback`` (issue #2901).

``Issues.are_issues_open`` reports a number the batched issue-state query could
not resolve once per *edge*: the number enters the baseline when a batch first
fails to resolve it and leaves only when a later batch positively resolves it.
Issue #1933 kept that baseline on the ``Issues`` instance. The fleet supervisor
builds a fresh ``GitHub`` per repo per pass (``fleet_dispatch.fleet_loop``), so
the instance baseline was empty at the start of every pass and "once" became
"every pass": 2,081 rows in a week in one consumer repo, one per
``loop_completed``.

The baseline therefore lives here, in a per-repo sidecar next to ``state.json``
(``layout.issue_state_unresolved_path``): the set of numbers that are
currently unresolved AND already reported. It follows the shape of the repo's
other small sidecars (``http_cache``, ``queue_sync_coverage_cache``): atomic
``write_json_atomic`` writes, a read-modify-write serialized by
``state.advisory_file_lock``, lock-free reads, and fail-safe handling of an
unusable file.

Why a sidecar and not ``state.json`` or ``events.db``:

* The writer is the GitHub client layer, which has no ``WriteGate`` and must
  not take the state lock (the ``write-gate-exempt`` note on the event emit).
* ``events.db`` is append-only and records no "resolved" edge, so deriving the
  baseline from it could never let a number that recovered and went stale
  again report afresh.

``settle_reported`` decides what is *fresh* inside the locked
read-modify-write, against the file as it is at that moment, so two processes
observing the same newly-unresolved number cannot both report it.

Fail-safe direction: telemetry must not be lost to a bookkeeping failure, so
an unreadable file reads as empty (every unresolved number reports once more
and the rewrite heals the file) and a failed write returns ``None``, which
tells the caller to fall back to its instance-local baseline -- exactly the
pre-#2901 behaviour. Nothing here raises into ``are_issues_open``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

from ..atomic_write import write_json_atomic
from ..state import StateLockBusy, advisory_file_lock, utc_now

logger = logging.getLogger(__name__)


def load_reported(path: Path) -> frozenset[int]:
    """The numbers already reported as unresolved. Lock-free.

    An atomic ``replace()`` never exposes a torn file to a lock-free reader.
    A missing, unreadable, or wrong-shaped file reads as empty (see the module
    docstring for why that is the safe direction).
    """
    if not path.exists():
        return frozenset()
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, LookupError, ValueError):
        logger.warning(
            "unresolved-issue baseline %s unreadable; treating as empty (every "
            "currently-unresolved number reports once more)",
            path,
        )
        return frozenset()
    numbers = data.get("unresolved") if isinstance(data, dict) else None
    if not isinstance(numbers, list):
        return frozenset()
    # ``bool`` is an ``int`` subclass; ``true`` is never an issue number.
    return frozenset(n for n in numbers if isinstance(n, int) and not isinstance(n, bool))


def settle_reported(
    path: Path, *, resolved: Iterable[int], unresolved: Iterable[int]
) -> frozenset[int] | None:
    """Apply one batch observation to the baseline; return the *fresh* numbers.

    The new baseline is ``(current - resolved) | unresolved``: a number a batch
    positively resolved leaves it (so a later re-stale is a fresh edge), a
    number the batch could not resolve joins it, and a number the batch never
    named is untouched -- an unrelated clean batch cannot reset the baseline.

    Returns the members of ``unresolved`` that were NOT already in the
    baseline: the edges to report. An empty result means "nothing to report",
    including the steady state of a number that stays unresolved. Returns
    ``None`` when the sidecar could not be updated; the caller then decides
    freshness from its own instance-local baseline.

    The steady state (nothing changes) costs one lock-free read and no write.
    """
    resolved_set = frozenset(resolved)
    unresolved_set = frozenset(unresolved)

    def advance(current: frozenset[int]) -> frozenset[int]:
        return (current - resolved_set) | unresolved_set

    try:
        # The lock-free read is inside the guard too: ``load_reported`` opens
        # with ``path.exists()``, which raises on errnos it does not treat as
        # "missing" (e.g. ``PermissionError``), and nothing here may raise into
        # ``are_issues_open``.
        current = load_reported(path)
        if advance(current) == current:
            return frozenset()
        path.parent.mkdir(parents=True, exist_ok=True)
        with advisory_file_lock(path):
            # Re-read under the lock: another process may have settled between
            # the lock-free read above and acquiring it.
            current = load_reported(path)
            updated = advance(current)
            if updated != current:
                write_json_atomic(
                    path,
                    {"version": 1, "generated_at": utc_now(), "unresolved": sorted(updated)},
                )
            return unresolved_set - current
    except (StateLockBusy, OSError) as exc:
        logger.warning(
            "unresolved-issue baseline write failed at %s; using the instance-local "
            "baseline for this observation (%s)",
            path,
            exc,
        )
        return None


__all__ = ["load_reported", "settle_reported"]
