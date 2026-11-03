"""Once-per-pass memo and durable change baseline for the worker selection answer (issue #2921).

``selection_wrapper.worker_selection`` decides, for each rendered worker or
rework prompt, whether the prompt may tell the worker to run ``ci-fleet test``.
It is called for every candidate on every pass, including candidates the
concurrency cap defers and never launches. Left alone that cost one ``ci-fleet
select`` subprocess and, whenever the answer was "no", one warning-level event
per prompt: 974 ``worker_test_selection_unavailable`` rows in one day in one
repo, all with the same reason, burying the warnings that mattered.

The answer is a property of the repo, not of the prompt, so this module holds
the two halves of "resolve it once and say it once":

* ``once_per_pass`` runs the resolution once per ``(pass, key)`` and hands the
  same value to every later prompt of that pass. A *pass* is the ambient
  correlation id: ``loop()`` wraps each pass in ``correlation_context()``, so
  nothing has to be reset and no caller has to remember to. With no
  correlation id there is no pass to scope a memo to, and the resolution runs
  every time -- exactly the behaviour before this module existed.
* ``settle_answer`` is the durable baseline: the last *unavailable* answer
  reported for the repo. The event is written only when the answer differs
  from it, and a usable wrapper clears it, so ``full`` -> usable -> ``full``
  reports the second ``full`` as the fresh edge it is.

Why the baseline is a sidecar (``layout.worker_selection_answer_path``) and not
an in-process attribute, ``state.json`` or ``events.db``: the fleet supervisor
rebuilds its per-repo objects every pass, so an in-process baseline is empty at
the start of every pass and "once" becomes "every pass" (the #2901 failure).
Prompts render outside the state lock, so ``state.json`` is not writable from
here. ``events.db`` is append-only and records no "usable again" edge, so it
could never tell a repeat from a recurrence. The sidecar follows
``github_capabilities.unresolved_issue_baseline``: atomic ``write_json_atomic``
writes, a read-modify-write serialized by ``state.advisory_file_lock``,
lock-free reads, and fail-safe handling of an unusable file.

Fail-safe direction: telemetry must not be lost to a bookkeeping failure. An
unreadable or missing file reads as "nothing reported" (an unavailable answer
reports once more and the rewrite heals the file); a failed write still
reports. Nothing here raises into a prompt render.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from pathlib import Path
from typing import TypeVar

from .atomic_write import write_json_atomic

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ``(reason, detail)`` of an unavailable answer. ``None`` is a usable wrapper.
Answer = tuple[str, str]

# Entries are keyed by correlation id, which never repeats, so an old entry is
# never *wrong* -- only dead weight. The bound keeps a long-lived supervisor
# from accumulating one entry per repo per pass forever.
MEMO_LIMIT = 256

_memo: OrderedDict[tuple[str, Hashable], object] = OrderedDict()
_memo_lock = threading.Lock()


def once_per_pass(key: Hashable, compute: Callable[[], T]) -> T:
    """``compute()`` once per ``(current pass, key)``; later calls in the pass reuse it.

    Outside a pass (no correlation id) nothing is memoized. ``compute`` runs
    outside the memo lock because it may spawn a subprocess; if two threads
    sharing one correlation id race on a key, both compute and the last write
    wins -- the sidecar lock in ``settle_answer`` still keeps their event to
    one row. An exception from ``compute`` propagates and memoizes nothing.
    """
    from .instrumentation import current_correlation_id

    correlation_id = current_correlation_id()
    if correlation_id is None:
        return compute()
    memo_key = (correlation_id, key)
    with _memo_lock:
        if memo_key in _memo:
            return _memo[memo_key]  # type: ignore[return-value]
    value = compute()
    with _memo_lock:
        _memo[memo_key] = value
        while len(_memo) > MEMO_LIMIT:
            _memo.popitem(last=False)
    return value


def load_answer(path: Path) -> Answer | None:
    """The last unavailable answer reported for the repo, or None. Lock-free.

    An atomic ``replace()`` never exposes a torn file to a lock-free reader. A
    missing, unreadable, or wrong-shaped file reads as None -- "nothing
    reported" -- which is the safe direction (see the module docstring).
    """
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, LookupError, ValueError):
        logger.warning(
            "worker-selection answer baseline %s unreadable; treating as nothing reported "
            "(an unavailable answer reports once more)",
            path,
        )
        return None
    recorded = data.get("answer") if isinstance(data, dict) else None
    if not isinstance(recorded, dict):
        return None
    reason, detail = recorded.get("reason"), recorded.get("detail")
    if not isinstance(reason, str) or not isinstance(detail, str):
        return None
    return reason, detail


def settle_answer(path: Path, answer: Answer | None) -> bool:
    """Record the repo's current answer; True when it is an unavailable answer to report.

    ``answer`` is ``(reason, detail)`` for an unavailable wrapper and ``None``
    for a usable one. The recorded baseline moves to ``answer`` whenever the two
    differ, and the return value is True only when ``answer`` is unavailable
    and differs from what was recorded: a repeat, and every usable answer,
    return False.

    The steady state (the answer is unchanged, usable or not) costs one
    lock-free read and no write. The comparison is repeated under the lock, so
    two processes settling the same new answer cannot both report it.
    """
    from .state import StateLockBusy, advisory_file_lock, utc_now

    try:
        if load_answer(path) == answer:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        with advisory_file_lock(path):
            # Re-read under the lock: another process may have settled between
            # the lock-free read above and acquiring it.
            if load_answer(path) == answer:
                return False
            write_json_atomic(
                path,
                {
                    "version": 1,
                    "generated_at": utc_now(),
                    "answer": None
                    if answer is None
                    else {"reason": answer[0], "detail": answer[1]},
                },
            )
            return answer is not None
    except (StateLockBusy, OSError) as exc:
        logger.warning(
            "worker-selection answer baseline write failed at %s; reporting this "
            "observation without a baseline (%s)",
            path,
            exc,
        )
        return answer is not None


__all__ = ["Answer", "MEMO_LIMIT", "load_answer", "once_per_pass", "settle_answer"]
