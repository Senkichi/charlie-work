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

This module owns the whole operator-stop control plane: the marker
primitives (``write_fleet_stop_request``/``read_fleet_stop_request``/
``clear_fleet_stop_request``), the ``fleet stop`` command body
(``run_fleet_stop``, re-exported through ``cli``), and the supervisor-side
machinery that honors a pending request (``FleetStopState`` for the
marker-poll/drain-latch inside ``run_fleet_supervise``,
``fleet_stop_pending`` for the ``supervise-loop`` relaunch gate,
``apply_fleet_drain_config``/``fleet_live_worker_count`` for drain passes,
and ``supervise_loop_interrupted_result`` for a clean Ctrl+C on the
wrapper).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import layout
from .fleet_paths import warn_fleet_dir_virtualization_on_write
from .instrumentation import log_event
from .process_utils import is_pid_alive
from .supervisor_lifecycle import read_supervisor_heartbeat, supervisor_heartbeat_path
from .workflow import CommandResult

if TYPE_CHECKING:
    from .config import OrchestratorConfig
    from .fleet_dispatch import FleetLocalSnapshot

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


def fleet_stop_pending(fleet_dir_override: str | None) -> bool:
    """Whether an operator stop/drain request is currently pending.

    The ``fleet supervise-loop`` relaunch gate uses this: a pending marker
    outranks a child's ``EXIT_RESTART_REQUESTED``, so a stop racing a
    self-deploy cannot bounce the supervisor back up.
    """
    return read_fleet_stop_request(fleet_dir_override) is not None


def fleet_live_worker_count(snapshot: FleetLocalSnapshot) -> int:
    """Total live workers across a fleet snapshot.

    The drain-exit signal for ``fleet stop --drain`` (issue #1716): the
    supervisor exits once this reaches zero. Read from the same config-aware
    per-repo session dirs the delta machinery already walks — no separate
    census, no hardcoded repo list.
    """
    return sum(repo_snap.live_count for _, repo_snap in snapshot.repo_snapshots)


def apply_fleet_drain_config(config: OrchestratorConfig) -> OrchestratorConfig:
    """Return the pass-local config copy ``fleet_loop`` uses while draining.

    Issue #1716: review dispatch is config-gated (``dispatch_reviews``
    early-returns after its reaper sweeps when the flag is off — those
    sweeps are exactly the cleanup a draining fleet still needs), so flip
    it off here. Fresh and rework dispatch are suppressed by the forced
    ``limit=0`` at the call sites.
    """
    return replace(
        config,
        review_dispatch=replace(config.review_dispatch, enabled=False),
    )


def _now_hms() -> str:
    return datetime.now().strftime("%H:%M:%S")


class FleetStopState:
    """The supervisor's operator stop/drain control-plane state (issue #1716).

    Polled every supervisor tick (i.e. between passes) — the only control
    surface a hidden scheduled supervisor has. A plain stop request exits at
    the next boundary with live workers untouched; a drain request latches
    ``draining`` and the pass launches nothing new. The marker is consumed
    only when honored, so a request written while no supervisor runs still
    takes effect on the next start. ``draining`` is latched rather than
    re-derived from the marker each tick so a torn marker rewrite mid-drain
    cannot silently re-arm dispatch.
    """

    def __init__(self) -> None:
        self._draining = False
        # Last live-worker count reported while draining; the "N remaining"
        # line prints only on change so a long drain doesn't spam the log.
        self._live_reported: int | None = None

    @property
    def draining(self) -> bool:
        return self._draining

    def poll_stop_marker(self, fleet_dir_override: str | None) -> str | None:
        """Poll the marker; latch drain or return the plain-stop exit reason.

        Returns ``"operator_stop"`` when a plain request was honored
        (marker consumed — the caller breaks out of the supervise loop);
        ``None`` when no request is pending or a drain request was latched.
        """
        request = read_fleet_stop_request(fleet_dir_override)
        if request is None:
            return None
        if request.get("drain"):
            if not self._draining:
                self._draining = True
                print(
                    f"[{_now_hms()}] operator drain requested: no new "
                    "dispatch; exiting when live workers reach 0",
                    flush=True,
                )
            return None
        clear_fleet_stop_request(fleet_dir_override)
        print(
            f"[{_now_hms()}] operator stop requested; exiting (live workers untouched)",
            flush=True,
        )
        return "operator_stop"

    def drain_tick(
        self,
        fleet_dir_override: str | None,
        snapshot: FleetLocalSnapshot,
        *,
        pass_due: bool,
    ) -> bool:
        """Report drain progress; return ``True`` once the drain is honored.

        Honors the drain only when nothing is in flight AND no pass is due:
        a due pass (e.g. the last worker's death IS the delta) runs first so
        its outcome is adopted before exit.
        """
        if not self._draining:
            return False
        live_workers = fleet_live_worker_count(snapshot)
        if live_workers != self._live_reported:
            self._live_reported = live_workers
            print(
                f"[{_now_hms()}] drain: {live_workers} live worker(s) remaining",
                flush=True,
            )
        if live_workers != 0 or pass_due:
            return False
        self._consume_and_announce(fleet_dir_override)
        return True

    def drain_complete_if_empty(
        self, fleet_dir_override: str | None, snapshot: FleetLocalSnapshot
    ) -> bool:
        """Post-pass variant: the due pass just adopted the last worker's
        outcome, so exit here rather than on the next poll tick."""
        if not self._draining or fleet_live_worker_count(snapshot) != 0:
            return False
        self._consume_and_announce(fleet_dir_override)
        return True

    def _consume_and_announce(self, fleet_dir_override: str | None) -> None:
        clear_fleet_stop_request(fleet_dir_override)
        print(f"[{_now_hms()}] drain complete: 0 live workers; exiting", flush=True)


def supervise_loop_interrupted_result() -> CommandResult:
    """The ``fleet supervise-loop`` wrapper's clean KeyboardInterrupt result.

    Issue #1716: the hidden scheduled deployment has no console, but a
    foreground ``fleet supervise-loop`` can still be interrupted — and
    interrupting the wrapper must not relaunch (the child keeps running
    detached either way). One clean line in the launcher log, no traceback;
    mirrors ``run_fleet_supervise``'s own KeyboardInterrupt -> clean-exit
    handling.
    """
    print("supervise-loop: interrupted; not relaunching", flush=True)
    return CommandResult(
        True,
        "supervise-loop: interrupted; not relaunching",
        {"interrupted": True},
    )


def run_fleet_stop(args: argparse.Namespace) -> CommandResult:
    """Write the operator stop/drain request marker for the fleet supervisor.

    Issue #1716: the only clean stop for a hidden scheduled fleet. The marker
    is the operative write; everything else here (audit event, liveness
    report, watchdog hint) is observability around it.
    """
    drain = bool(getattr(args, "drain", False))
    marker_path = layout.fleet_stop_request_path(args.fleet_dir)

    if args.dry_run:
        return CommandResult(
            True,
            f"dry-run: would write {'drain' if drain else 'stop'} request to {marker_path}",
            {"dry_run": True, "drain": drain, "marker": str(marker_path)},
        )

    prior = read_fleet_stop_request(args.fleet_dir)
    # write_fleet_stop_request also records the fleet_stop_requested audit
    # event (best-effort) so every writer — not just this command — is traced.
    path = write_fleet_stop_request(args.fleet_dir, drain=drain)

    heartbeat = read_supervisor_heartbeat(args.fleet_dir)
    supervisor_live = bool(
        heartbeat
        and not heartbeat.get("exited_at")
        and isinstance(heartbeat.get("pid"), int)
        and is_pid_alive(heartbeat["pid"])
    )

    # The marker is consumed when honored, so an armed charlie-fleet-pass
    # trigger relaunches a clean supervisor on its next tick — the trap the
    # runbook's "disable the task first" procedure exists for. Report the
    # probe's actual answer rather than restating the doc.
    # Function-local import: ``fleet_dispatch`` imports this module, so a
    # module-level import would be circular. Resolving at call time also
    # means tests patch the name on ``fleet_dispatch`` itself — there is no
    # fleet_stop-local binding to shadow.
    from .fleet_dispatch import probe_fleet_watchdog

    watchdog = probe_fleet_watchdog()
    if watchdog.armed is True:
        watchdog_hint = (
            "the charlie-fleet-pass task is armed and relaunches the fleet on "
            "its next tick; disable the task first if the fleet must stay down"
        )
    elif watchdog.armed is False:
        watchdog_hint = (
            "the charlie-fleet-pass task is disabled; the fleet stays down "
            "until it is re-enabled or supervise is started manually"
        )
    else:
        watchdog_hint = (
            "could not probe the charlie-fleet-pass task; if enabled, its "
            "trigger relaunches the fleet on its next tick"
        )

    if drain:
        action = "the supervisor dispatches nothing new and exits when live workers reach 0"
    else:
        action = "the supervisor exits at its next pass boundary (live workers untouched)"
    message = f"stop request recorded at {path}: {action}"
    if not supervisor_live:
        message += "; no live supervisor detected — the request takes effect on the next start"
    message += f"; {watchdog_hint}"

    return CommandResult(
        True,
        message,
        {
            "marker": str(path),
            "drain": drain,
            "replaced_prior_request": prior is not None,
            "supervisor_live": supervisor_live,
            "watchdog_armed": watchdog.armed,
            "watchdog_detail": watchdog.detail,
        },
    )
