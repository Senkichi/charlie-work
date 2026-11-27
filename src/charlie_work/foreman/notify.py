"""Item notification: send each unnotified open item through the notify sinks.

Design §8.3 puts ``item store, then notify`` on the ``raise_item`` path; this
is the notify half. Each open item whose ``notified_at`` is empty goes out as
a generic ``NotifyMessage`` through the hub's configured sink; a successful
send stamps ``notified_at`` so the item is never sent twice, and a failed
send leaves it unstamped for the next pass. A disabled notify layer sends
and stamps nothing — enabling it later still delivers the backlog.

Nothing calls this yet (issue #3121): the Turn's Apply step wires it in when
the foreman pass lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..notify import NotifyMessage, emit_message
from . import Result
from .items import (
    ITEM_REPORT,
    Item,
    _load_items_strict,
    mark_notified,
    open_items,
)


@dataclass(frozen=True)
class ItemNotifyOutcome:
    """What one notify pass did: stamped item ids and per-item send failures."""

    sent: tuple[str, ...] = ()
    failed: tuple[tuple[str, str], ...] = ()  # (item id, sink or stamp error)


def _message(foreman_root: Path, item: Item) -> NotifyMessage:
    """The generic message for one item: title, body, and the remedy command."""
    name = foreman_root.name
    if item.kind == ITEM_REPORT:
        body = item.text or ""
    else:
        lines = [item.question or ""]
        for index, option in enumerate(item.options):
            marker = " (recommended)" if index == item.recommended else ""
            lines.append(f"{index}. {option}{marker}")
        body = "\n".join(line for line in lines if line)
    command = item.command
    if command is None and item.kind != ITEM_REPORT:
        choice = item.recommended if item.recommended is not None else "<index>"
        command = f"charlie gang intervene {name} {item.id} --choose {choice}"
    return NotifyMessage(
        title=f"foreman {name}: {item.kind} {item.id}", body=body, command=command
    )


def notify_unnotified_items(
    foreman_root: Path, config: Any, *, at: str | None = None
) -> Result[ItemNotifyOutcome]:
    """Send every open item with no ``notified_at`` and stamp the delivered ones.

    Errors-as-values end to end: a corrupt ``items.json`` is a ``Result.error``
    (the strict read, because this pass also *writes* the stamps), a disabled
    notify layer is a quiet empty outcome, and a per-item sink or stamp
    failure lands in ``outcome.failed`` without stopping the rest.
    """
    loaded = _load_items_strict(foreman_root)
    if not loaded.ok:
        return Result(error=loaded.error)
    if not getattr(config, "enabled", False):
        return Result(value=ItemNotifyOutcome())
    sent: list[str] = []
    failed: list[tuple[str, str]] = []
    for item in open_items(loaded.value):
        if item.notified_at:
            continue
        delivered = emit_message(config, _message(foreman_root, item))
        if not delivered.ok:
            failed.append((item.id, delivered.error or "sink failed"))
            continue
        stamped = mark_notified(foreman_root, item.id, at=at)
        if not stamped.ok:
            # Sent but unstamped: the next pass sends it again — a duplicate
            # notification beats a silent one, but the failure is recorded.
            failed.append((item.id, f"sent but not stamped: {stamped.error}"))
            continue
        sent.append(item.id)
    return Result(value=ItemNotifyOutcome(sent=tuple(sent), failed=tuple(failed)))


__all__ = ["ItemNotifyOutcome", "notify_unnotified_items"]
