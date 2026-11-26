"""The sanctioned lock-free, non-mutating state-file reader (issue #3141).

``state.load_state`` must only run under ``state_lock``: on read/parse
failure it calls ``_quarantine_state``, which RENAMES the state file -- the
issue #310 hazard for any reader that samples a concurrently-written file
(a racing tmp+replace reads as corrupt, and the healthy file gets renamed
away). ``read_state_snapshot`` is the non-mutating sibling for callers that
must not take the lock at all: the fleet-wide live census (which walks
other repos' state under the fleet launch lock) and best-effort pre-checks
(the launch-gate throttle check, the operator-marker liveness probe, the
closing-keyword autofix attempt counter). It never quarantines, renames, or
writes anything, and it reports failure instead of returning an empty
state.

File-size ratchet (#1442): the implementation lives here rather than in
``state.py`` (which is over its high-water mark); ``state.py`` re-exports
the symbol so ``state.read_state_snapshot`` is the stable import path.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_state_snapshot(path: Path) -> dict[str, Any] | None:
    """Lock-free, non-mutating read of a state file (issue #3141).

    The single sanctioned unlocked state reader: ``load_state`` must only run
    under ``state_lock`` (its ``_quarantine_state`` rename makes an unlocked
    read of a concurrently-written file the issue #310 hazard — a racing
    tmp+replace reads as corrupt and the healthy file gets renamed away).
    This primitive NEVER mutates: no quarantine rename, no write of any kind.
    State writes are atomic replaces, so a lock-free read observes the last
    committed snapshot — which is all a fleet-wide census or a best-effort
    pre-check needs.

    Returns ``empty_state()`` when ``path`` does not exist (a repo that has
    not run legitimately has no dispatch records), the parsed and normalized
    state otherwise, and ``None`` on read or parse failure after the same
    bounded OSError retries ``load_state`` uses. ``None`` means "state
    unavailable" — the caller decides (skip with a loud signal, fail closed,
    retry later); it must never be treated as an empty state.
    """
    from . import state as _state  # lazy: state.py re-exports this symbol

    if not path.exists():
        return _state.empty_state()

    data: Any = None
    for attempt in range(_state._LOAD_RETRY_ATTEMPTS):
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, LookupError, ValueError) as exc:
            # Corrupt state (truncation, wrong encoding) — a signal, never a
            # rename: quarantining here could rename a healthy file whose
            # tmp+replace merely raced this read (issue #310).
            logger.warning(
                "read_state_snapshot: %s is unreadable (%s: %s)",
                path,
                type(exc).__name__,
                exc,
            )
            return None
        except OSError as exc:
            # Sharing/permission violations on Windows are often transient.
            # Retry before reporting the state as unavailable.
            if attempt < _state._LOAD_RETRY_ATTEMPTS - 1:
                time.sleep(_state._LOAD_RETRY_DELAY_SECONDS)
                continue
            logger.warning(
                "read_state_snapshot: %s could not be read after %d attempts (%s: %s)",
                path,
                _state._LOAD_RETRY_ATTEMPTS,
                type(exc).__name__,
                exc,
            )
            return None
        else:
            break

    if not isinstance(data, dict):
        return None
    data.setdefault("version", _state.STATE_VERSION)
    data.setdefault("generated_at", _state.utc_now())
    data.setdefault("issues", {})
    data.setdefault("prs", {})
    data.setdefault("events", [])
    data.setdefault("throttled_until", None)
    return data
