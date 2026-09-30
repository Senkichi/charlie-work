"""Leaf parser for ``state.json`` ISO 8601 timestamps.

Single point of enforcement for a parser that used to be hand-duplicated in
``workflow.py`` and ``worker_fate.py``. This module imports nothing from the
package, so ``state``, ``workflow`` and ``worker_fate`` can all import it
without creating a cycle. ``state.parse_iso_timestamp`` re-exports it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def parse_iso_timestamp(value: Any) -> datetime | None:
    """Parse an ISO 8601 timestamp into an aware ``datetime``.

    Naive results are assumed UTC, matching every writer in this codebase.
    Non-string, non-datetime and unparseable values return ``None``.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
