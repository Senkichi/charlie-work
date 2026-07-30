"""Fleet supervisor lifecycle instrumentation (issue #627).

Records ``supervisor_started`` / ``supervisor_exited`` events to the
fleet-level ``events.db`` and maintains a ``supervisor-heartbeat.json``
sidecar so an abnormal termination (``TerminateProcess`` / ``kill -9``)
leaves a diagnosable gap instead of only a launcher text marker.

Design notes:

- **The heartbeat file is the ground truth that survives a kill.** The
  supervisor writes it on startup and every loop iteration; on a clean
  exit it stamps ``exited_at`` / ``exit_code``. A killed supervisor
  leaves ``exited_at`` null, so the next start detects the gap and emits
  a retroactive ``supervisor_exited`` event. ``TerminateProcess`` skips
  Python ``finally`` blocks, so the supervisor itself can never record
  its own abnormal exit — only the next start, or the independent
  ``scripts/heartbeat_check.py`` freshness check, can.
- **Events go to ``<fleet_dir>/events.db``.** ``log_event`` derives the
  database path as ``state_path.parent / "events.db"``, so passing the
  heartbeat file as ``state_path`` lands the events alongside it.
  Supervisor lifecycle is a fleet-level concern; per-repo ``events.db``
  files do not see it (and should not — one supervisor, many repos).
- **``supervisor_exited`` is alertable when ``exit_code`` is not ``0``.**
  A ``0`` exit is a routine drain / HEAD-drift / self-deploy restart;
  anything else means something killed it. A gap-detected exit has
  ``exit_code=None`` and is alertable. This mirrors the ``alertable``
  distinction added in #621 (a preview failure must not page, a real
  one must).

All filesystem writes use the project's atomic temp-file + ``replace()``
discipline. Event logging is best-effort and never raises —
instrumentation must never break the supervisor's exit path.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .fleet_paths import fleet_dir, warn_fleet_dir_virtualization_on_write
from .instrumentation import log_event

logger = logging.getLogger(__name__)

HEARTBEAT_FILENAME = "supervisor-heartbeat.json"

# Event kind strings. Recorded in ``events.db`` and surfaced to the
# attention digest on abnormal exits.
SUPERVISOR_STARTED = "supervisor_started"
SUPERVISOR_EXITED = "supervisor_exited"

#: ``repo`` field stamped on fleet-level supervisor events. Matches the
#: ``repo="fleet"`` used by the fleet attention digest.
_FLEET_REPO = "fleet"


def supervisor_heartbeat_path(fleet_dir_override: str | None = None) -> Path:
    """Return the path to the supervisor heartbeat sidecar.

    Lives in the fleet directory alongside ``fleet.json`` and the
    fleet-level ``events.db``. The heartbeat file doubles as the
    ``state_path`` passed to ``log_event`` so its sibling ``events.db``
    is the fleet-level event database.
    """
    return fleet_dir(override=fleet_dir_override) / HEARTBEAT_FILENAME


def _read_heartbeat(path: Path) -> dict[str, Any] | None:
    """Read the heartbeat sidecar, returning ``None`` if missing or unreadable.

    A corrupt or unparseable file is non-fatal: the caller treats
    ``None`` as "no prior supervisor state", which is the conservative
    reading (no retroactive exit event, fresh heartbeat written).
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_heartbeat(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist the heartbeat sidecar (temp-file + ``replace()``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    warn_fleet_dir_virtualization_on_write(
        path.parent, context="writing supervisor-heartbeat.json"
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _seconds_between(start_iso: str | None, end_iso: str | None) -> float | None:
    """Return the elapsed seconds between two ISO timestamps, or ``None``."""
    start = _parse_iso(start_iso)
    end = _parse_iso(end_iso)
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds())


def write_supervisor_heartbeat(path: Path, payload: dict[str, Any]) -> None:
    """Write the heartbeat sidecar (thin public wrapper over :func:`_write_heartbeat`)."""
    _write_heartbeat(path, payload)


def read_supervisor_heartbeat(fleet_dir_override: str | None = None) -> dict[str, Any] | None:
    """Read the current heartbeat sidecar (public wrapper over :func:`_read_heartbeat`)."""
    return _read_heartbeat(supervisor_heartbeat_path(fleet_dir_override))


def record_supervisor_started(
    fleet_dir_override: str | None,
    *,
    pid: int,
    started_at: str,
    full_pass_interval_seconds: int,
    max_pass_runtime_seconds: int | None = None,
) -> None:
    """Emit ``supervisor_started`` and write a fresh heartbeat.

    Called once after the supervisor lock is acquired — acquiring the
    lock proves any prior supervisor is gone (byte-range locks are
    released on process death), so a stale heartbeat with no
    ``exited_at`` here means the prior supervisor was killed, not that
    it is still running.
    """
    if max_pass_runtime_seconds is None:
        max_pass_runtime_seconds = full_pass_interval_seconds
    path = supervisor_heartbeat_path(fleet_dir_override)
    heartbeat = {
        "pid": pid,
        "started_at": started_at,
        "last_beat_at": started_at,
        "pass_number": 0,
        "full_pass_interval_seconds": full_pass_interval_seconds,
        "max_pass_runtime_seconds": max_pass_runtime_seconds,
        "exited_at": None,
        "exit_code": None,
    }
    _write_heartbeat(path, heartbeat)
    log_event(
        path,
        SUPERVISOR_STARTED,
        {
            "pid": pid,
            "started_at": started_at,
            "full_pass_interval_seconds": full_pass_interval_seconds,
            "max_pass_runtime_seconds": max_pass_runtime_seconds,
        },
        repo=_FLEET_REPO,
    )


def update_supervisor_heartbeat(
    fleet_dir_override: str | None,
    *,
    pass_number: int,
    last_beat_at: str,
) -> None:
    """Refresh ``last_beat_at`` / ``pass_number`` on the existing heartbeat.

    Called at the top of every supervisor loop iteration so the
    freshness signal stays tight (at most one ``poll_interval_seconds``
    old on a live supervisor). Preserves ``started_at`` / ``pid`` /
    ``exited_at`` / ``exit_code`` from the existing file. Best-effort:
    a write failure is logged and swallowed so a disk hiccup does not
    abort the pass.
    """
    path = supervisor_heartbeat_path(fleet_dir_override)
    existing = _read_heartbeat(path) or {}
    existing.update(
        {
            "last_beat_at": last_beat_at,
            "pass_number": pass_number,
        }
    )
    try:
        _write_heartbeat(path, existing)
    except OSError as exc:
        logger.warning("Failed to refresh supervisor heartbeat at %s: %s", path, exc)


def detect_prior_abnormal_exit(
    fleet_dir_override: str | None,
) -> dict[str, Any] | None:
    """Detect that a prior supervisor terminated without recording an exit.

    Returns a payload describing the prior supervisor when the heartbeat
    sidecar exists, has a ``last_beat_at``, and has no ``exited_at``
    (i.e. the previous supervisor never reached its clean-exit path —
    it was killed or hung). Returns ``None`` when there is no prior
    supervisor, the prior one exited cleanly, or the heartbeat is
    unreadable (conservative: cannot prove an abnormal exit).

    The caller emits a retroactive ``supervisor_exited`` event from
    this payload and alerts on it.
    """
    data = _read_heartbeat(supervisor_heartbeat_path(fleet_dir_override))
    if data is None:
        return None
    if data.get("exited_at"):
        # A clean exit was recorded — the prior supervisor's own exit
        # path already emitted its ``supervisor_exited`` event.
        return None
    if not data.get("last_beat_at"):
        return None
    return {
        "prior_pid": data.get("pid"),
        "prior_started_at": data.get("started_at"),
        "prior_last_beat_at": data.get("last_beat_at"),
        "prior_pass_number": data.get("pass_number"),
        "uptime_seconds": _seconds_between(data.get("started_at"), data.get("last_beat_at")),
    }


def record_prior_abnormal_exit(
    fleet_dir_override: str | None,
    prior: dict[str, Any],
) -> dict[str, Any]:
    """Emit a retroactive ``supervisor_exited`` event for a killed prior supervisor.

    ``exit_code`` is ``None`` (the cause is unknown by construction —
    that is the point of issue #627) so the event is alertable. Returns
    the event payload so the caller can also route it to the attention
    digest.
    """
    payload = {
        "exit_code": None,
        "uptime_seconds": prior.get("uptime_seconds"),
        "passes": prior.get("prior_pass_number"),
        "reason": "prior_supervisor_terminated_without_exit_event",
        "prior_pid": prior.get("prior_pid"),
        "prior_started_at": prior.get("prior_started_at"),
        "prior_last_beat_at": prior.get("prior_last_beat_at"),
    }
    log_event(
        supervisor_heartbeat_path(fleet_dir_override),
        SUPERVISOR_EXITED,
        payload,
        repo=_FLEET_REPO,
    )
    return payload


def record_supervisor_exit(
    fleet_dir_override: str | None,
    *,
    exit_code: int,
    passes: int,
    started_at: str,
    reason: str,
) -> dict[str, Any]:
    """Emit ``supervisor_exited`` and stamp the heartbeat with the exit.

    Called from the supervisor's ``finally`` block on every in-control
    exit (drain, max_runtime, max_passes, HEAD-drift restart,
    self-deploy restart, ``KeyboardInterrupt``, or an uncaught
    exception). A ``TerminateProcess`` kill skips this entirely — the
    heartbeat keeps ``exited_at=null`` and the next start detects the
    gap via :func:`detect_prior_abnormal_exit`.

    Returns the event payload. Best-effort: a heartbeat write failure
    is logged and swallowed so the exit path never raises.
    """
    path = supervisor_heartbeat_path(fleet_dir_override)
    exited_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload = {
        "exit_code": exit_code,
        "uptime_seconds": _seconds_between(started_at, exited_at),
        "passes": passes,
        "reason": reason,
    }
    log_event(path, SUPERVISOR_EXITED, payload, repo=_FLEET_REPO)
    existing = _read_heartbeat(path) or {}
    existing.update({"exited_at": exited_at, "exit_code": exit_code})
    try:
        _write_heartbeat(path, existing)
    except OSError as exc:
        logger.warning("Failed to stamp supervisor exit in heartbeat at %s: %s", path, exc)
    return payload


def is_exit_alertable(exit_code: int | None) -> bool:
    """Return True when a supervisor exit warrants an operator-facing alert.

    A ``0`` exit is routine (drain, HEAD-drift restart, self-deploy
    restart, ``KeyboardInterrupt``). Anything else — a non-zero exit or
    an unknown ``None`` (gap-detected kill) — means something killed
    the supervisor and must reach the attention digest. This is the
    #621 ``alertable`` precedent applied to supervisor lifecycle: a
    routine restart must not page, a real kill must.
    """
    return exit_code != 0
