"""Foreman item reads for the Now collector (read-only; issue #3121, ADR-0008).

The needs-me zone lists open ``decision``/``permission`` items for every
foreman. The dashboard reads only ``<fleet_dir>/foremen/<name>/items.json``
— never writes, never calls GitHub, and never imports ``charlie_work.foreman``
(the import-direction guard in ``tests/test_foreman_import_direction.py`` is
structural, so the file shape is decoded locally and tolerantly here: a
malformed record is skipped, a missing file is empty).

Open means what the item store means: ``open``, or ``snoozed`` whose
``snoozed_until`` has passed (an unparseable deadline counts as expired —
fail-visible, matching the store's reader).
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ..safe_path import contains
from .now_types import ForemanItemRead
from .sources import _parse_utc, read_json_file

# Same names as ``foreman/paths.py`` — spelled here for the same reason
# ``sources.RUNNER_ALLOCATION_FILENAME`` is: the package boundary is a
# feature, so the read side cannot import the layout constants it shadows.
FOREMEN_DIRNAME = "foremen"
ITEMS_FILENAME = "items.json"

#: Same pattern as ``foreman.paths.FOREMAN_NAME_RE`` — and the same reason
#: ``_shared`` can never be mistaken for a foreman directory.
_FOREMAN_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]*")

_NEEDS_KINDS = frozenset({"decision", "permission"})


_STATUSES = frozenset({"open", "snoozed", "answered", "dismissed"})


def _record_open(record: dict[str, Any], now: datetime) -> bool:
    """The store's open semantics on a raw record: ``open`` always counts;
    ``snoozed`` counts once its deadline has passed (or cannot be parsed)."""
    status = record.get("status")
    if status not in _STATUSES:
        # The store reader decodes an unknown or absent status as ``open``.
        status = "open"
    if status == "open":
        return True
    if status != "snoozed":
        return False
    until = _parse_utc(record.get("snoozed_until"))
    return until is None or until <= now


def _decode(foreman: str, record: Any, now: datetime) -> ForemanItemRead | None:
    if not isinstance(record, dict):
        return None
    kind = record.get("kind")
    if kind not in _NEEDS_KINDS or not _record_open(record, now):
        return None
    item_id = record.get("id")
    if not isinstance(item_id, str):
        return None
    question = record.get("question")
    text = record.get("text")
    options = record.get("options")
    option_count = len(options) if isinstance(options, list) else 0
    recommended = record.get("recommended")
    if not isinstance(recommended, int) or not 0 <= recommended < option_count:
        recommended = None
    command = record.get("command")
    return ForemanItemRead(
        foreman=foreman,
        item_id=item_id,
        kind=kind,
        text=question if isinstance(question, str) else (text if isinstance(text, str) else ""),
        command=command if isinstance(command, str) else None,
        recommended=recommended,
        created_at=_parse_utc(record.get("created_at")),
    )


def read_foreman_items(fleet_dir: Path, now: datetime) -> tuple[ForemanItemRead, ...]:
    """Every open decision/permission item across ``<fleet_dir>/foremen/*/``.

    Tolerant end to end: a foreman directory with a missing or corrupt
    ``items.json`` contributes nothing, and one bad record never hides the
    items beside it.
    """
    root = Path(fleet_dir) / FOREMEN_DIRNAME
    try:
        entries = sorted(root.iterdir(), key=lambda p: p.name)
    except OSError:
        return ()
    out: list[ForemanItemRead] = []
    for entry in entries:
        if not entry.is_dir() or not _FOREMAN_NAME_RE.fullmatch(entry.name):
            continue
        if not contains(root, entry):
            # A reparse point landing outside foremen/ is foreign, not a foreman.
            continue
        read = read_json_file(entry / ITEMS_FILENAME)
        records = read.data.get("items") if read.data is not None else None
        if not isinstance(records, list):
            continue
        for record in records:
            decoded = _decode(entry.name, record, now)
            if decoded is not None:
                out.append(decoded)
    return tuple(out)


__all__ = ["read_foreman_items"]
