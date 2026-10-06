"""Worker worktrees and uv cache on the ci-fleet host I/O volume.

A host provisioned for fast I/O carries a ci-fleet manifest (schema 1) whose
``worker_root``/``worker_uv_cache`` name a trusted volume reserved for fleet
workers. When the manifest is valid and its volume is mounted, the default
worktrees root moves to ``<worker_root>/<repo name>/worktrees`` and every
worker launch gets ``UV_CACHE_DIR=<worker_uv_cache>``, so ``uv`` can hardlink
from cache to venv on one volume. ``TMP``/``TEMP`` need nothing here: the
per-session temp dir (issue #1767, ``layout.worker_tmp_dir``) lives inside the
worktree, so it moves with it.

Precedence for the worktrees root: an explicit ``claude_code.worktrees_dir`` >
the manifest > the ``runtime.state_dir`` default. ``claude_code.host_io_worktrees:
false`` is the kill switch. Any manifest or volume problem falls back to the
default root for that resolution; nothing here raises.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ci_fleet.host_io.manifest import (
    Absent,
    HostIo,
    VolumeProbe,
    load_manifest,
    probe_volume,
    resolve_manifest_path,
    volume_ok,
)

from . import layout
from .atomic_write import write_json_atomic

logger = logging.getLogger(__name__)

#: ``<fleet_dir>/host-io-worker-cutover.json``: when worker worktrees first moved
#: onto the volume. ci-fleet's worker A/B reads it to split before/after windows.
CUTOVER_FILENAME = "host-io-worker-cutover.json"
CUTOVER_SCHEMA = 1


@dataclass(frozen=True)
class WorkerIo:
    """The active host I/O locations for fleet workers."""

    worker_root: Path
    uv_cache: Path
    drive: str


@dataclass(frozen=True)
class WorkerIoStatus:
    """``io`` when active; ``note`` says why not (empty when there is no manifest)."""

    io: WorkerIo | None
    note: str


#: The status when nothing has been resolved (the ``ResolvedLayout`` default).
NO_WORKER_IO = WorkerIoStatus(None, "")


def resolve_worker_io(
    *,
    enabled: bool,
    manifest: str = "",
    probe: VolumeProbe | None = None,
) -> WorkerIoStatus:
    """Read the manifest and check its volume. Never raises."""
    if not enabled:
        return WorkerIoStatus(None, "disabled by claude_code.host_io_worktrees")
    parsed = load_manifest(resolve_manifest_path(manifest))
    if isinstance(parsed, Absent):
        return WorkerIoStatus(None, "")
    if not isinstance(parsed, HostIo):
        note = f"host-io manifest {parsed.path} is invalid: {parsed.reason}"
        logger.warning("%s; worker worktrees stay on the default root", note)
        return WorkerIoStatus(None, note)
    ok, reason = volume_ok(parsed, probe=probe or probe_volume)
    if not ok:
        note = f"host-io volume unavailable: {reason}"
        logger.warning("%s; worker worktrees stay on the default root", note)
        return WorkerIoStatus(None, note)
    return WorkerIoStatus(
        WorkerIo(Path(parsed.worker_root), Path(parsed.worker_uv_cache), parsed.drive), ""
    )


def worktrees_root(io: WorkerIo, repo_root: Path) -> Path:
    """``<worker_root>/<repo name>/worktrees``."""
    return layout.worktrees_dir(io.worker_root / repo_root.name)


def worker_env(io: WorkerIo | None) -> dict[str, str]:
    """The env layer for worker launches; operator ``worker_env`` is merged over it."""
    return {} if io is None else {"UV_CACHE_DIR": str(io.uv_cache)}


def record_cutover(
    fleet_dir: Path, io: WorkerIo | None, *, dry_run: bool, now: datetime | None = None
) -> bool:
    """Write the cutover marker once, the first time a live dispatch uses the volume.

    Best-effort: it runs on every adapter-settings build, so a write failure
    is logged and never fails the dispatch.
    """
    if io is None or dry_run:
        return False
    marker = fleet_dir / CUTOVER_FILENAME
    at = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        if marker.exists():
            return False
        marker.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(marker, {"schema": CUTOVER_SCHEMA, "cutover_at": at})
    except OSError as exc:
        logger.warning("could not write host-io cutover marker %s: %s", marker, exc)
        return False
    return True
