"""Exact id cursor for the ``events`` table (issue #3114).

Extracted out of ``instrumentation.py`` under the file-size ratchet (issue
#1442): the #3114 feature's ``query_events(after_id=...)``/``_row_to_event``
edits are issue-mandated in place, but ``max_event_id`` is new code with no
mandated home, and landing it in ``instrumentation.py`` grew a file already
over the 800-line cap. So the function moves here and is re-exported through
``instrumentation.py``'s facade import block (the same ``from .X import (...)
# noqa: F401 (deliberate re-export)`` pattern the #1283 Phase-A extraction
lineage established). Callers keep importing it from
``charlie_work.instrumentation`` unchanged.

The private helper this calls (``_get_db``) stays in ``instrumentation.py`` --
it is shared by every read/write primitive there -- and is imported lazily
inside the function to avoid a top-level import cycle (``instrumentation``
re-exports this module's ``max_event_id``). The lazy ``from .instrumentation
import ...`` form also resolves ``_get_db`` as a module attribute at call
time, so tests patching ``instrumentation._get_db`` keep working. The lazy
pattern is already used at ``base_branch.py`` and ``workflow.py:5076``.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def max_event_id(state_path: Path) -> int:
    """Return the largest ``id`` in the ``events`` table, or 0.

    A new ``after_id`` consumer starts its cursor here to mean "now":
    ``query_events(after_id=max_event_id(path))`` then yields every event
    written from that point on. Returns 0 for an empty table, a missing or
    unreadable database (best-effort, never raises), so a cursor of 0 pages
    the whole log from the first row.
    """
    # Lazy import: ``_get_db`` lives in ``instrumentation.py`` (shared by
    # every read/write primitive there), and ``instrumentation`` re-exports
    # this function -- a top-level mutual import would cycle.
    from .instrumentation import _get_db

    conn = _get_db(state_path)
    if conn is None:
        return 0
    try:
        row = conn.execute("SELECT MAX(id) FROM events").fetchone()
    except sqlite3.Error as exc:
        logger.warning("Failed to read max event id: %s", exc)
        return 0
    return int(row[0]) if row is not None and row[0] is not None else 0
