"""Read-only source layer for the fleet dashboard (ADR-0008, spec sections 2/5/6/6a).

Everything the dashboard shows is read from files and SQLite databases the
orchestrator already writes; nothing here writes fleet state or calls GitHub.
Two invariants hold for every function in this module:

* **Registry-only enumeration.** Repos come from ``fleet.json`` and nowhere
  else. Globbing for ``events.db`` would also hit per-worktree and test-scratch
  databases (events recon section 2), so a path that is not in the registry is
  never opened.
* **Errors come back as values.** A missing, corrupt, or locked source yields a
  result object with ``error`` set; nothing raises, so one dead lane cannot blank
  the whole dashboard.

This module must not import ``ci_fleet`` actuating modules (``runner_slots``,
``runners``, ``runner_allocation_pass``): importing them puts park/start/terminate
one call away from a read-only page. ``tests/test_dashboard_sources.py`` pins that
via the module AST. The runner-allocation sidecar is therefore addressed by its
file name here rather than through ``ci_fleet.runner_slots.ALLOCATION_STATE_FILENAME``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import layout

# Private on purpose: no public events-DB path helper exists (`_db_path` is the one derivation
# log_event itself uses, and cli/doctor/fleet_status import it too); promote it in a follow-up.
from ..instrumentation import _db_path
from ..supervisor_lifecycle import HEARTBEAT_FILENAME

# Same file name as ``ci_fleet.runner_slots.ALLOCATION_STATE_FILENAME``; spelled
# here to avoid importing that actuating module (see module docstring).
RUNNER_ALLOCATION_FILENAME = "runner-allocation.json"


@dataclass(frozen=True)
class RepoSource:
    """One registered repo's readable sources (resolved from ``fleet.json``)."""

    key: str
    repo_root: Path
    state_dir: Path
    snapshot_path: Path
    events_db: Path


@dataclass(frozen=True)
class FleetSources:
    """Host-wide (fleet dir) sources, all addressed through layout helpers."""

    fleet_dir: Path
    registry: Path
    events_db: Path
    supervisor_heartbeat: Path
    runner_allocation: Path
    capacity_starvation_state: Path
    fleet_pause: Path


@dataclass(frozen=True)
class SnapshotRead:
    """Result of reading one ``status-snapshot.json`` envelope."""

    written_at: datetime | None
    age_seconds: float | None
    data: dict[str, Any] | None
    error: str | None


@dataclass(frozen=True)
class JsonRead:
    """Result of reading a small JSON sidecar file."""

    data: dict[str, Any] | None
    error: str | None


def fleet_sources(fleet_dir_override: str | None = None) -> FleetSources:
    """Resolve the global fleet-dir sources (honours ``CHARLIE_WORK_FLEET_DIR``)."""
    root = layout.fleet_dir(override=fleet_dir_override)
    heartbeat = root / HEARTBEAT_FILENAME
    return FleetSources(
        fleet_dir=root,
        registry=layout.fleet_registry_path(override=fleet_dir_override),
        # The global events.db sits beside the heartbeat: log_event derives it as
        # ``state_path.parent / "events.db"`` (supervisor_lifecycle.py).
        events_db=_db_path(heartbeat),
        supervisor_heartbeat=heartbeat,
        runner_allocation=root / RUNNER_ALLOCATION_FILENAME,
        capacity_starvation_state=layout.capacity_starvation_state_path(
            override=fleet_dir_override
        ),
        fleet_pause=layout.fleet_pause_path(override=fleet_dir_override),
    )


def read_registry(path: Path) -> tuple[dict[str, Any], str | None]:
    """Parse ``fleet.json``; a missing file is an empty registry, a bad one is an error.

    ``fleet_registry._load_registry`` swallows corruption into an empty registry, which
    would make an unreadable file look like "no repos registered". Here the two differ:
    the second element is an error string only when the file exists but cannot be used.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"repos": {}}, None
    except OSError as exc:
        return {"repos": {}}, f"fleet registry unreadable: {path}: {exc}"
    try:
        data = json.loads(text)
    except ValueError as exc:
        return {"repos": {}}, f"fleet registry corrupt: {path}: {exc}"
    if not isinstance(data, dict) or not isinstance(data.get("repos", {}), dict):
        return {"repos": {}}, f"fleet registry corrupt: {path}: unexpected structure"
    return data, None


def load_repos(
    fleet_dir_override: str | None = None,
) -> tuple[tuple[RepoSource, ...], str | None]:
    """``enumerate_repos`` plus the registry error (None when readable or absent)."""
    data, error = read_registry(layout.fleet_registry_path(override=fleet_dir_override))
    out: list[RepoSource] = []
    for key, entry in sorted(data.get("repos", {}).items()):
        if not isinstance(entry, dict) or not entry.get("repo_root") or not entry.get("state_dir"):
            continue
        state_dir = Path(str(entry["state_dir"]))
        out.append(
            RepoSource(
                key=key,
                repo_root=Path(str(entry["repo_root"])),
                state_dir=state_dir,
                snapshot_path=layout.status_snapshot_path(state_dir),
                events_db=_db_path(layout.state_file_path(state_dir)),
            )
        )
    return tuple(out), error


def enumerate_repos(fleet_dir_override: str | None = None) -> tuple[RepoSource, ...]:
    """Return one ``RepoSource`` per ``fleet.json`` entry, sorted by key.

    Entries without a ``repo_root`` or ``state_dir`` cannot be located and are
    skipped; paths are not checked for existence (that is the reader's error
    value, so a vanished lane stays visible as an error rather than disappearing).
    A missing or corrupt registry yields ``()``; callers that must tell those apart
    use ``load_repos``.
    """
    return load_repos(fleet_dir_override)[0]


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def read_json_file(path: Path) -> JsonRead:
    """Read a JSON object file; every failure is an ``error`` string."""
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            loaded = json.load(handle)
    except FileNotFoundError:
        return JsonRead(None, f"missing: {path}")
    except (OSError, ValueError) as exc:
        return JsonRead(None, f"unreadable: {path}: {exc}")
    if not isinstance(loaded, dict):
        return JsonRead(None, f"not a JSON object: {path}")
    return JsonRead(loaded, None)


def read_snapshot(repo: RepoSource, now: datetime) -> SnapshotRead:
    """Read ``status-snapshot.json`` without touching GitHub.

    Freshness is ``now - envelope.snapshot_written_at``: the ``snapshot_written_at``
    and ``cache_age_seconds`` keys inside ``data`` are always null on disk (snapshot
    recon section 1), so they are never consulted. ``now`` is injected so callers
    and tests control the clock. The writer replaces the file atomically, so a
    lock-free read never sees a torn file.
    """
    raw = read_json_file(repo.snapshot_path)
    if raw.data is None:
        return SnapshotRead(None, None, None, raw.error)
    written_at = _parse_utc(raw.data.get("snapshot_written_at"))
    if written_at is None:
        return SnapshotRead(
            None, None, None, f"no valid snapshot_written_at: {repo.snapshot_path}"
        )
    data = raw.data.get("data")
    if not isinstance(data, dict):
        return SnapshotRead(
            written_at, None, None, f"snapshot has no data object: {repo.snapshot_path}"
        )
    return SnapshotRead(written_at, (now - written_at).total_seconds(), data, None)


def open_events_ro(path: Path) -> tuple[sqlite3.Connection | None, str | None]:
    """Open an ``events.db`` read-only; returns ``(conn, error)`` (exactly one is None).

    Uses a ``file:...?mode=ro`` URI so the open can neither create the file nor
    take a write lock against the supervisor's WAL writer. A missing file is
    reported as a value instead of letting sqlite create an empty database.

    Unlike ``metrics_base.open_dashboard_ro`` this deliberately does NOT pass
    ``immutable=1``: ``events.db`` is appended continuously by a live writer and its
    WAL is not checkpointed per event, so an immutable reader would silently miss
    every row still in the WAL. The ``-shm``/``-wal`` sidecars a plain read-only
    open may create next to the database are the accepted price of seeing fresh rows.
    """
    if not path.is_file():
        return None, f"missing: {path}"
    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")  # fail fast on a non-database
    except sqlite3.Error as exc:
        return None, f"cannot open {path}: {exc}"
    return conn, None


def latest_event(conn: sqlite3.Connection, kind: str) -> dict[str, Any] | None:
    """Return the newest row of ``kind`` as a dict (payload JSON-decoded), or None.

    None means "no such row or unreadable"; callers needing to tell those apart
    must use ``open_events_ro`` first (this never raises).
    """
    try:
        row = conn.execute(
            "SELECT id, ts, kind, payload, repo, correlation_id, pr_number, issue_number, level"
            " FROM events WHERE kind = ? ORDER BY id DESC LIMIT 1",
            (kind,),
        ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    event = dict(row)
    try:
        event["payload"] = json.loads(event["payload"]) if event["payload"] else {}
    except ValueError:
        event["payload"] = {}
    return event
