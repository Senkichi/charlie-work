"""Deferred ``uv sync`` marker and its starvation bound (issue #1855).

Extracted from ``supervise.py`` so new code does not land in an over-cap
monolith (file-size ratchet, issue #1442). This module owns the
pending-sync marker schema (read/write/clear under
``layout.pending_sync_path``), the episode ``written_at`` timestamp the
marker carries, the age computation the starvation bound measures, the
default bound, and the once-per-episode ``self_deploy_sync_starved``
event emission. ``supervise.py`` re-exports this surface so existing
imports and tests keep working.

Self-deploy defers ``uv sync`` while fleet workers are live. Under
continuous load that deferral was unbounded (issue #1855). The marker's
``written_at`` records the FIRST deferral of an episode and is carried
forward verbatim on every rewrite, so its age bounds the whole episode --
not the latest pass. Once the age crosses ``starvation_seconds`` the
deferred result reports ``starved=True`` and the caller drains new
dispatches so the live-worker count reaches zero and the sync can land;
nothing is killed. A single ``self_deploy_sync_starved`` event fires per
episode, deduplicated by the ``starved_notified`` latch in the marker
itself so the signal survives the head-moved supervisor restarts that
happen mid-episode.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import layout
from .atomic_write import write_json_atomic
from .instrumentation import log_event
from .state import utc_now
from .subprocess_runner import RunResult, command_failure_message, run_captured

#: Event kind for the deferred-sync starvation signal (issue #1855). Read
#: here (not re-declared at call sites) for the same reason label strings
#: are read from ``LabelConfig``: ``doctor_sync_starvation`` queries this
#: kind back out of events.db, and a literal scattered across emit/query
#: sites drifts silently.
SELF_DEPLOY_SYNC_STARVED_KIND = "self_deploy_sync_starved"

#: Default starvation bound (seconds) for a deferred ``uv sync`` that stays
#: pending under continuous live fleet workers before the supervisor stops
#: admitting new dispatches (issue #1855). Mirrors
#: ``config.FleetSupervisorConfig.dependency_sync_starvation_seconds``'s default;
#: <= 0 disables the bound.
DEFAULT_SYNC_STARVATION_SECONDS = 14400


@dataclass(frozen=True)
class SyncDeferral:
    """Outcome of recording one deferred pass of a pending-sync episode.

    ``starved`` is True when the episode's age -- measured from the
    marker's ``written_at`` -- has crossed the configured bound this pass,
    including passes where the ``starved_notified`` latch was already set.
    ``marker_age_seconds`` is ``None`` when the marker has no usable
    ``written_at`` (a pre-#1855 marker or a hand-edit).
    """

    starved: bool
    marker_age_seconds: float | None


def _pending_sync_marker_path(state_root: Path) -> Path:
    """Return the path to the deferred-``uv sync`` marker under ``state_root``."""
    return layout.pending_sync_path(state_root)


def _write_marker(
    path: Path, from_sha: str, to_sha: str, *, starved_notified: bool = False
) -> None:
    """Persist the pending-sync marker atomically (temp-file + replace).

    ``written_at`` records the FIRST deferral of this episode and is carried
    forward verbatim on every later rewrite (each deferred pass refreshes
    ``to_sha`` as new pulls land) -- it is what the starvation bound
    (issue #1855) measures, and it must survive the head-moved supervisor
    restarts that happen mid-episode. A missing or unparseable prior
    ``written_at`` re-arms to now (a pre-#1855 marker, or a hand-edit, gets
    one fresh window rather than an unverifiable age). Same shape for
    ``starved_notified``: latched true once the ``self_deploy_sync_starved``
    event fires, preserved across rewrites, so the event fires once per
    episode rather than once per deferred pass.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = _read_marker(path)
    prior_written = prior.get("written_at")
    payload: dict[str, Any] = {
        "from_sha": from_sha,
        "to_sha": to_sha,
        "written_at": (
            prior_written if _parse_marker_timestamp(prior_written) is not None else utc_now()
        ),
    }
    if starved_notified or prior.get("starved_notified"):
        payload["starved_notified"] = True
    write_json_atomic(path, payload)


def _read_marker(path: Path) -> dict[str, Any]:
    """Read the marker, returning an empty dict on any read/parse error."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    # A valid-JSON non-mapping marker (hand-edit, torn schema change) reads
    # as absent rather than crashing the deferral branch on ``marker.get``.
    return data if isinstance(data, dict) else {}


def _parse_marker_timestamp(value: Any) -> datetime.datetime | None:
    """Parse a marker ``written_at`` ISO-8601 string, or ``None`` on any miss.

    Naive timestamps are read as UTC -- the writer (``utc_now``) always emits
    ``Z``, so a naive value can only come from a hand-edited marker, and
    assuming UTC errs toward a *larger* measured age (the starvation-safe
    direction) when the local convention was UTC anyway.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.UTC)
    return parsed


def _pending_sync_age_seconds(
    marker: dict[str, Any] | None, *, now: datetime.datetime
) -> float | None:
    """Age of the pending-sync episode in seconds, or ``None`` when unknown.

    ``None`` covers an absent marker and a marker with no usable
    ``written_at`` -- the safe direction is to keep deferring on an
    unverifiable age rather than drain-dispatch on one.
    """
    if not marker:
        return None
    start = _parse_marker_timestamp(marker.get("written_at"))
    if start is None:
        return None
    return (now - start).total_seconds()


def _clear_marker(path: Path) -> None:
    """Remove the pending-sync marker, if it exists."""
    path.unlink(missing_ok=True)


def record_sync_deferral(
    marker_path: Path,
    prior_marker: dict[str, Any] | None,
    *,
    from_sha: str,
    to_sha: str,
    live_count: int,
    starvation_seconds: int,
    state_path: Path,
    now: datetime.datetime | None = None,
) -> SyncDeferral:
    """Persist one deferred pass and emit the once-per-episode starvation event.

    Bounds how long a pending sync may starve under continuous load (issue
    #1855). The marker's ``written_at`` is the FIRST deferral of this
    episode (``_write_marker`` carries it forward), so the bound measures
    the whole episode, not the latest pass. ``starved`` reports the trip to
    the caller -- which stops admitting new dispatches -- rather than
    killing anything live, the same non-destructive posture as the rest of
    the self-deploy path.

    ``prior_marker`` is the marker read before this pass's rewrite (or
    ``None`` when absent); the rewrite happens unconditionally so a fresh
    ``to_sha`` lands even on a starved pass. When the bound has tripped and
    the latch is unset, one ``self_deploy_sync_starved`` event is emitted to
    ``state_path``'s events.db and ``starved_notified`` is latched --
    emission precedes the latch so a crash between the two re-emits on the
    next pass, the safe direction for a once-per-episode signal.
    ``starvation_seconds <= 0`` disables the bound.
    """
    marker_age = _pending_sync_age_seconds(
        prior_marker, now=now or datetime.datetime.now(datetime.UTC)
    )
    starved = (
        starvation_seconds > 0 and marker_age is not None and marker_age >= starvation_seconds
    )
    _write_marker(marker_path, from_sha, to_sha)
    if starved and not (prior_marker or {}).get("starved_notified"):
        # write-gate-exempt(issue=1855): durable events.db signal, outside state lock
        log_event(
            state_path,
            SELF_DEPLOY_SYNC_STARVED_KIND,
            {
                "pending_seconds": int(marker_age or 0),
                "starvation_seconds": starvation_seconds,
                "live_count": live_count,
                "from_sha": from_sha,
                "to_sha": to_sha,
            },
        )
        _write_marker(marker_path, from_sha, to_sha, starved_notified=True)
    return SyncDeferral(starved=starved, marker_age_seconds=marker_age)


#: Event kind for the boot-time pending-sync backstop (issue #2312). One
#: event per repair attempt that actually detected skew -- keyed by
#: ``detected_via`` so marker-replay and probe-replay paths are observable
#: separately. A clean no-op emits nothing.
SELF_DEPLOY_BOOT_SYNC_KIND = "self_deploy_boot_sync"

_BOOT_SYNC_PROBE_TIMEOUT_SECONDS = 30
_BOOT_SYNC_SYNC_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class BootSyncRepair:
    """Outcome of a single ``heal_pending_sync_at_boot`` pass.

    ``detected_via`` is ``"marker"`` (a pending-sync marker whose ``to_sha``
    equals HEAD), ``"probe"`` (``uv sync --locked --check --inexact`` reported
    a clean "would change" exit),
    or ``None`` when nothing needed repair. ``synced`` is True only after a
    successful ``uv sync --locked --inexact``. ``deferred`` is True when skew was
    detected but live fleet runners are active -- the merge and sync stay
    parked for ``self_deploy``/the next restart. ``detail`` carries the
    operator-facing reason for deferred/failed outcomes.
    """

    detected_via: str | None = None
    synced: bool = False
    deferred: bool = False
    detail: str | None = None


def heal_pending_sync_at_boot(
    repo_root: Path,
    *,
    state_root: Path,
    live_count: Callable[[], int],
    run_command: Callable[..., RunResult] = run_captured,
    probe_timeout_seconds: int = _BOOT_SYNC_PROBE_TIMEOUT_SECONDS,
    sync_timeout_seconds: int = _BOOT_SYNC_SYNC_TIMEOUT_SECONDS,
) -> BootSyncRepair:
    """Repair a skewed orchestrator venv before the supervisor loads config.

    Startup backstop for issue #2312: a pre-#2312 deferral -- or any
    crash-ordering survivor -- can leave a marker replaying (HEAD already at
    the marker's ``to_sha``) while the venv still holds the old commit's
    dependencies. If the next ``fleet supervise`` launch hits config
    validation in that state, the supervisor bricks before ``self_deploy``
    ever gets a pass to retry. This runs *before* config loading on every
    supervisor entrypoint and:

    1. detects skew via a marker that is replaying (marker present and HEAD
       equals its ``to_sha``) or via ``uv sync --locked --check --inexact``
       (marker lost/corrupt and a ``uv.lock`` exists -- that is precisely the
       bricking scenario; a uv-free dev checkout without a lockfile is not).
       Only the probe's clean "would change" exit (1) counts as drift --
       a uv failure, timeout, missing binary, or unsupported ``--check``
       reports detail instead of triggering a sync -- and ``--inexact``
       keeps extraneous packages (e.g. the documented ``uv sync
       --all-extras`` dev environment) from ever reading as drift,
    2. defers when live fleet runners are active (same live-runner rule as
       ``self_deploy``: do not rebuild a venv running children use),
    3. otherwise runs ``uv sync --locked --inexact`` and clears the marker
       on success.

    Every terminal state -- detected+deferred, detected+synced,
    detected+sync-failed, probe-inconclusive, and crashed -- emits a
    ``self_deploy_boot_sync`` event. A clean no-op detects nothing and
    emits nothing.

    Never raises and never writes code: failures return
    ``BootSyncRepair(detail=...)`` and the caller decides whether to log or
    ignore. A failed sync leaves the marker in place so the next restart
    retries.

    The caller (``run_fleet_supervise``) invokes this only while holding
    the fleet-supervisor lock, so the probe and the sync can never run
    concurrently with a live supervisor's own ``uv sync``.
    """

    marker_path = layout.pending_sync_path(state_root)
    state_path = layout.state_file_path(state_root)
    try:
        return _boot_sync_repair(
            repo_root,
            marker_path=marker_path,
            state_path=state_path,
            live_count=live_count,
            run_command=run_command,
            probe_timeout_seconds=probe_timeout_seconds,
            sync_timeout_seconds=sync_timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 -- startup backstop; never break boot
        detail = f"{type(exc).__name__}: {exc}"
        try:
            # write-gate-exempt(issue=2312): durable events.db signal, outside state lock
            log_event(
                state_path,
                SELF_DEPLOY_BOOT_SYNC_KIND,
                {
                    "detected_via": None,
                    "deferred": False,
                    "synced": False,
                    "error": detail,
                },
            )
        except Exception:  # noqa: BLE001 -- even the state path may be unwritable
            pass
        return BootSyncRepair(detail=detail)


def _boot_sync_repair(
    repo_root: Path,
    *,
    marker_path: Path,
    state_path: Path,
    live_count: Callable[[], int],
    run_command: Callable[..., RunResult],
    probe_timeout_seconds: int,
    sync_timeout_seconds: int,
) -> BootSyncRepair:
    marker = _read_marker(marker_path) if marker_path.exists() else None
    detected_via: str | None = None
    if marker:
        head_res = run_command(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            timeout_seconds=probe_timeout_seconds,
        )
        to_sha = marker.get("to_sha")
        if head_res.ok and to_sha and head_res.stdout.strip() == to_sha:
            # Marker replaying = a pre-#2312 deferral that moved HEAD before
            # parking the sync. A marker whose to_sha is still ahead of HEAD
            # is an in-flight #2312 deferral; self_deploy owns its merge.
            detected_via = "marker"
    if detected_via is None and (repo_root / "uv.lock").exists():
        # --inexact: extraneous packages are not drift. The documented dev
        # environment is ``uv sync --all-extras``, whose pytest/ruff/pluggy
        # sit outside the default dependency set -- an exact-mode probe
        # reports "would uninstall" on every idle boot, and the repair below
        # would prune the dev extras out of the very venv running it
        # (observed: a mid-suite heal pruned pluggy and every spawned pytest
        # child died on import). The repair only ever needs to *add* missing
        # locked packages, so preserving installed extras is correct on the
        # probe and the sync alike.
        probe_cmd = ["uv", "sync", "--locked", "--check", "--inexact"]
        probe = run_command(
            probe_cmd,
            cwd=repo_root,
            timeout_seconds=probe_timeout_seconds,
        )
        if probe.returncode == 1:
            detected_via = "probe"
        elif probe.returncode != 0:
            # Only a clean "would change" exit (1) is drift. Anything else --
            # a spawn failure or timeout (returncode None), or a uv error
            # like an unparseable lockfile or an unsupported ``--check``
            # (2+) -- proves nothing about the venv, and a blind sync under a
            # broken probe is exactly the false-positive this command exists
            # to prevent. Report it instead of syncing.
            detail = command_failure_message(probe_cmd, probe, "uv sync --check failed")
            # write-gate-exempt(issue=2312): durable events.db signal, outside state lock
            log_event(
                state_path,
                SELF_DEPLOY_BOOT_SYNC_KIND,
                {
                    "detected_via": None,
                    "deferred": False,
                    "synced": False,
                    "error": detail,
                },
            )
            return BootSyncRepair(detail=detail)
    if detected_via is None:
        # Nothing skewed. A marker in this state holds its merge (to_sha is
        # still ahead of HEAD); self_deploy owns it on the next pass.
        return BootSyncRepair()

    live = live_count()
    if live > 0:
        # write-gate-exempt(issue=2312): durable events.db signal, outside state lock
        log_event(
            state_path,
            SELF_DEPLOY_BOOT_SYNC_KIND,
            {
                "detected_via": detected_via,
                "deferred": True,
                "synced": False,
                "live_count": live,
            },
        )
        return BootSyncRepair(
            detected_via=detected_via,
            deferred=True,
            detail=f"{live} live fleet runner(s) still active",
        )

    sync_cmd = ["uv", "sync", "--locked", "--inexact"]
    sync_res = run_command(
        sync_cmd,
        cwd=repo_root,
        timeout_seconds=sync_timeout_seconds,
    )
    if not sync_res.ok:
        detail = command_failure_message(sync_cmd, sync_res, "uv sync failed")
        # write-gate-exempt(issue=2312): durable events.db signal, outside state lock
        log_event(
            state_path,
            SELF_DEPLOY_BOOT_SYNC_KIND,
            {
                "detected_via": detected_via,
                "deferred": False,
                "synced": False,
                "error": detail,
            },
        )
        return BootSyncRepair(detected_via=detected_via, detail=detail)

    cleared = False
    if marker:
        _clear_marker(marker_path)
        cleared = True
    # write-gate-exempt(issue=2312): durable events.db signal, outside state lock
    log_event(
        state_path,
        SELF_DEPLOY_BOOT_SYNC_KIND,
        {
            "detected_via": detected_via,
            "deferred": False,
            "synced": True,
            "cleared_marker": cleared,
        },
    )
    return BootSyncRepair(detected_via=detected_via, synced=True)
