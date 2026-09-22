"""Operator stop/drain request marker for the fleet supervisor (issue #1716).

The scheduled-task deployment (``charlie-fleet-pass``) runs the supervisor
hidden — ``wscript -> powershell -> cmd -> uv -> supervise-loop -> supervise`` —
so there is no console to send ``Ctrl+C`` to. ``charlie fleet stop [--drain]``
writes a JSON marker into the fleet dir instead:

- ``fleet supervise`` reads it between passes. A plain request exits at the
  next poll boundary with workers untouched; a ``drain`` request suppresses
  new dispatch and exits once the live-worker count reaches zero.
- ``fleet supervise-loop`` reads it when its child asks to be replaced
  (``EXIT_RESTART_REQUESTED``): a pending stop wins over the relaunch, so an
  operator stop racing a self-deploy cannot bounce the supervisor back up.

Lifecycle: the marker is consumed (deleted) by the supervisor that honors it.
An unhonored marker persists by design — a request written while no
supervisor runs takes effect on the next start rather than being silently
dropped. After a plain stop the marker is gone, so the scheduled task's next
tick can relaunch a clean supervisor (the runbook tells operators to disable
the task first when the fleet must stay down).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import layout
from .fleet_paths import warn_fleet_dir_virtualization_on_write
from .instrumentation import log_event
from .supervisor_lifecycle import supervisor_heartbeat_path

logger = logging.getLogger(__name__)

#: events.db kind recorded when ``charlie fleet stop`` writes the marker.
#: Registered at ``info`` in ``instrumentation._LEVEL_BY_KIND`` — an
#: operator-initiated request, not an anomaly.
FLEET_STOP_REQUESTED = "fleet_stop_requested"


def write_fleet_stop_request(fleet_dir_override: str | None, *, drain: bool) -> Path:
    """Atomically write the stop-request marker and return its path.

    Temp-file + ``replace()`` so a reader never sees a half-written marker —
    the same atomic-write pattern as every other JSON state file. The last
    writer wins: ``fleet stop`` then ``fleet stop --drain`` upgrades the
    pending request to a drain, and the reverse de-escalates it.

    Every write is dual-recorded as a fleet-level ``fleet_stop_requested``
    event in events.db (anchored on the supervisor-heartbeat state path the
    lifecycle events already use). Audit is best-effort: its failure must
    never mask a marker that landed.
    """
    path = layout.fleet_stop_request_path(fleet_dir_override)
    warn_fleet_dir_virtualization_on_write(
        path, context=f"writing {layout.FLEET_STOP_REQUEST_FILENAME}"
    )
    replaced_prior = read_fleet_stop_request(fleet_dir_override) is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "fleet_stop_request",
        "requested_at": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "drain": drain,
        "requester_pid": os.getpid(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    try:
        log_event(
            supervisor_heartbeat_path(fleet_dir_override),
            FLEET_STOP_REQUESTED,
            {
                "drain": drain,
                "marker": str(path),
                "replaced_prior_request": replaced_prior,
            },
            repo="fleet",
        )
    except Exception:  # noqa: BLE001 - audit must not mask the recorded write
        logger.warning("could not record %s event", FLEET_STOP_REQUESTED, exc_info=True)
    return path


def read_fleet_stop_request(
    fleet_dir_override: str | None,
) -> dict[str, Any] | None:
    """Return the pending stop request payload, or ``None``.

    ``None`` covers absent, unreadable, and malformed markers alike — a
    corrupt file must not wedge the supervisor (the operator can re-run
    ``fleet stop``, which overwrites atomically).
    """
    path = layout.fleet_stop_request_path(fleet_dir_override)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        logger.warning("unreadable fleet stop-request marker %s; ignoring", path)
        return None
    if not isinstance(data, dict):
        return None
    return data


def clear_fleet_stop_request(fleet_dir_override: str | None) -> bool:
    """Consume the marker. Returns ``True`` when one was removed."""
    path = layout.fleet_stop_request_path(fleet_dir_override)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError:
        logger.warning("could not remove fleet stop-request marker %s", path)
        return False
    return True
