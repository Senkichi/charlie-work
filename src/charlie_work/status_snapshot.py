"""Status-snapshot cache for ``fleet status --json`` (issue #1463).

Extracted from ``workflow.py`` per the #1442 file-size ratchet's prescribed
remedy (domain module + facade re-export through ``workflow.py``). The loop
pass writes ``status()``'s result to ``status-snapshot.json`` at the end of
every pass (atomic temp-file + ``replace``); ``status()`` serves from that
snapshot when it is fresher than ``runtime.status_snapshot_ttl_seconds``,
turning a ~50s serial GitHub API walk into a sub-second file read on an idle
host and eliminating the state-lock contention that pushed wall time past the
heartbeat's 60s cap during concurrent loop passes.

The read is lock-free: the write uses atomic temp-file + ``replace``, so a
reader either sees the previous complete snapshot or no file at all, never a
partial write.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from . import layout
from .state import utc_now

if TYPE_CHECKING:
    # ``CommandResult`` and ``OrchestratorApp`` live in ``workflow.py``; the
    # annotations are strings (``from __future__ import annotations``), so
    # these are only needed for type-checkers, not at runtime. The runtime
    # ``CommandResult`` construction in ``read_status_snapshot`` uses a
    # deferred import to avoid the circular dependency.
    from .workflow import CommandResult, OrchestratorApp

_LOG = logging.getLogger(__name__)


def snapshot_path(paths_root: Path) -> Path:
    """Return the per-repo status-snapshot cache path under ``paths_root``."""
    return layout.status_snapshot_path(paths_root)


def read_status_snapshot(app: OrchestratorApp) -> CommandResult | None:
    """Return a cached ``status()`` result if the snapshot is fresh.

    Returns ``None`` when the snapshot is absent, stale, or unreadable —
    the caller falls back to a live computation in all those cases. The
    read is lock-free: ``write_status_snapshot`` uses atomic temp-file +
    ``replace``, so this reader either sees the previous complete snapshot
    or no file at all, never a partial write (issue #1463).
    """
    ttl = app.config.runtime.status_snapshot_ttl_seconds
    if ttl <= 0:
        return None
    path = snapshot_path(app.paths.root)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            envelope = json.load(handle)
    except (json.JSONDecodeError, OSError, ValueError):
        # Corrupt or transiently-unreadable snapshot — fall back to live.
        return None
    if not isinstance(envelope, dict):
        return None
    written_at = envelope.get("snapshot_written_at")
    if not isinstance(written_at, str):
        return None
    try:
        written_ts = datetime.fromisoformat(written_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    age = (datetime.now(UTC) - written_ts).total_seconds()
    if age > ttl:
        return None
    data = envelope.get("data")
    if not isinstance(data, dict):
        return None
    # Surface cache freshness into the ``data`` dict itself, not just
    # ``CommandResult.message`` (which ``run_fleet_status`` discards).
    # ``snapshot_written_at`` / ``cache_age_seconds`` let ``heartbeat_check``
    # and human operators distinguish a fresh live response (both ``None``)
    # from a cached one (timestamp + age). Injected here rather than stored
    # in the envelope's ``data`` so the written snapshot never carries a
    # stale freshness field of its own.
    data = dict(data)
    data["snapshot_written_at"] = written_at
    data["cache_age_seconds"] = round(age, 1)
    from .workflow import CommandResult  # deferred: avoid circular import

    return CommandResult(True, "status complete (cached)", data)


def write_status_snapshot(app: OrchestratorApp) -> None:
    """Write a fresh ``status()`` snapshot for ``fleet status`` to serve.

    Called at the end of every loop pass. Computes the live status (bypassing
    the cache) and writes it atomically so a concurrent ``fleet status --json``
    invocation never sees a partial file. Best-effort: failures are logged but
    never propagate — a failed snapshot write must not crash the loop pass
    (issue #1463).
    """
    try:
        result = app.status(use_cache=False)
        if not result.ok:
            return
        envelope = {
            "snapshot_written_at": utc_now(),
            "data": result.data,
        }
        path = snapshot_path(app.paths.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(envelope, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(path)
    except Exception:
        _LOG.warning("status snapshot write failed for %s", app.repo_root, exc_info=True)
