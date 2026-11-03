"""First-class fleet pause/resume (issue #1776).

``charlie fleet pause [--reason TEXT]`` writes a persistent ``fleet-pause.json``
flag into the fleet dir; ``charlie fleet resume`` removes it. It replaces the
hand-run kill procedure (disable task, wait for a boundary, kill an 8-process
chain by PID without a tree kill, prove it stayed down):

- ``fleet supervise`` reads the flag fresh at the top of each loop iteration --
  between passes, never mid-pass -- and exits with ``EXIT_FLEET_PAUSED`` after
  stamping ``exit_code``/``exited_at`` on the heartbeat. Nothing is killed:
  listeners and in-flight workers keep running.
- ``fleet supervise-loop`` lets a pending pause outrank a child's restart
  request, like a pending stop marker.
- ``scripts/fleet-pass.ps1`` refuses to start while the flag exists, so the
  scheduled task stays enabled and a pause survives reboot/logon.
- The one-shot pass entrypoints (``fleet work|bash-rats``, ``work``,
  ``bash-rats``) are gated once, in ``cli.main``, before ``self_deploy()`` or
  any dispatch (issue #2660): they exit ``EXIT_FLEET_PAUSED`` unless given
  ``--ignore-pause``. A command opts in via ``mark_pass_running``.
- A pause also stops *new worker launches* inside a fleet pass already
  running (issue #2766): a fleet pass walks the registry for minutes, so the
  loop-boundary read above would otherwise let every repo it has not reached
  launch a full cap of workers after the operator paused. The one gate every
  worker launch goes through (``worker_launch_gate.issue_worker_launch_permit``)
  reads the flag fresh via :func:`launch_pause_fields` and defers with
  ``fleet_paused``. ``--ignore-pause`` rides on the app
  (:func:`ignore_pause_requested`) so a deliberate one-shot still launches.

Unlike the one-shot stop marker (``fleet_stop``) the flag is never consumed by
the supervisor; only ``resume`` removes it. The flag is read from disk every
iteration -- config is cached at supervisor startup, so a config knob is no
substitute.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from . import host as _host
from . import layout
from .atomic_write import write_json_atomic
from .fleet_paths import warn_fleet_dir_virtualization_on_write
from .supervisor_lifecycle import read_supervisor_heartbeat, supervisor_heartbeat_path
from .command_result import CommandResult
from .write_gate import WriteGate

logger = logging.getLogger(__name__)

#: events.db kinds recorded by pause/resume (registered ``info`` in
#: ``event_levels/``).
FLEET_PAUSED = "fleet_paused"
FLEET_RESUMED = "fleet_resumed"

#: Printed by ``pause`` and surfaced by ``status`` while paused: the pause
#: freezes the runner-allocation pass too, and an unnoticed pause is an outage.
PAUSE_CONSEQUENCE = (
    "runner allocation is driven by the same supervisor pass, so CI capacity "
    "stays frozen at the last converged parked floor until `charlie fleet resume`"
)


def _fleet_write_gate(fleet_dir_override: str | None) -> WriteGate:
    """Live (non-dry-run) gate for fleet-level audit events.

    Dry-run callers return before reaching a write, so the gate is always live.
    """
    return WriteGate(
        dry_run=False,
        state_path=supervisor_heartbeat_path(fleet_dir_override),
        repo="fleet",
    )


def write_fleet_pause(fleet_dir_override: str | None, *, reason: str | None = None) -> Path:
    """Atomically write the pause flag (temp + ``replace()``) and return its path.

    Re-pausing overwrites: the last reason wins. The audit event is
    best-effort and must never mask a flag that landed.
    """
    path = layout.fleet_pause_path(fleet_dir_override)
    warn_fleet_dir_virtualization_on_write(path, context=f"writing {layout.FLEET_PAUSE_FILENAME}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "fleet_pause",
        "paused_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "reason": reason,
        "requester_pid": os.getpid(),
    }
    write_json_atomic(path, payload)
    try:
        write_gate = _fleet_write_gate(fleet_dir_override)
        write_gate.log_event(kind=FLEET_PAUSED, payload={"flag": str(path), "reason": reason})
    except Exception:  # noqa: BLE001 - audit must not mask the recorded write
        logger.warning("could not record %s event", FLEET_PAUSED, exc_info=True)
    return path


def read_fleet_pause(fleet_dir_override: str | None) -> dict[str, Any] | None:
    """Return the pause payload, or ``None`` when not paused.

    Existence is the signal: a present-but-unreadable/malformed flag still
    means *paused* (payload ``{}``) -- failing open would silently resume a
    fleet the operator paused. Only an absent file is ``None``. "Malformed"
    includes bytes that are not UTF-8: Windows PowerShell 5.1 writes UTF-16
    for ``... > fleet-pause.json``, and since issue #2766 this read sits on
    every worker launch, where raising would crash the lane instead of
    deferring it.
    """
    path = layout.fleet_pause_path(fleet_dir_override)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError):
        logger.warning("unreadable fleet pause flag %s; treating as paused", path)
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("malformed fleet pause flag %s; treating as paused", path)
        return {}
    return data if isinstance(data, dict) else {}


def fleet_pause_pending(fleet_dir_override: str | None) -> bool:
    """Whether the pause flag exists (read fresh from disk every call)."""
    return read_fleet_pause(fleet_dir_override) is not None


def clear_fleet_pause(fleet_dir_override: str | None) -> bool:
    """Remove the flag. Returns ``True`` when one was removed."""
    path = layout.fleet_pause_path(fleet_dir_override)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def fleet_pause_status(fleet_dir_override: str | None) -> dict[str, Any] | None:
    """Status-report block for a paused fleet, or ``None`` when running."""
    flag = read_fleet_pause(fleet_dir_override)
    if flag is None:
        return None
    return {
        "paused": True,
        "paused_at": flag.get("paused_at"),
        "reason": flag.get("reason"),
        "flag": str(layout.fleet_pause_path(fleet_dir_override)),
        "consequence": PAUSE_CONSEQUENCE,
    }


#: argparse default a pass-running command carries (set by ``mark_pass_running``).
#: ``cli.main`` gates on it, so the pause check has one enforcement point and a
#: command cannot be pass-running without also being guarded (issue #2660).
PASS_RUNNING_DEST = "runs_pass"


def mark_pass_running(parser: argparse.ArgumentParser) -> None:
    """Declare ``parser``'s command pass-running (issue #2660).

    One call does both halves, so neither can be forgotten alone: it sets the
    ``runs_pass`` default ``refuse_pass_while_paused`` keys on, and adds the
    ``--ignore-pause`` override that gate honours.
    """
    parser.add_argument(
        "--ignore-pause",
        action="store_true",
        default=False,
        dest="ignore_pause",
        help=(
            f"Run even though the fleet is paused ({layout.FLEET_PAUSE_FILENAME} present, #1776). "
            "For a deliberate operator one-shot; logged as an override."
        ),
    )
    parser.set_defaults(**{PASS_RUNNING_DEST: True})


def ignore_pause_requested(args: argparse.Namespace) -> bool:
    """Whether ``--ignore-pause`` was given (``False`` for a command without the flag).

    The one reader of the flag's dest, so ``refuse_pass_while_paused`` (which
    lets the command start) and the app constructors (which let it launch) cannot
    disagree about what the override means.
    """
    return bool(getattr(args, "ignore_pause", False))


class PauseAwareApp(Protocol):
    """What the launch gate reads off an ``OrchestratorApp``."""

    fleet_dir_override: str | None
    ignore_pause: bool


def launch_pause_fields(app: PauseAwareApp) -> dict[str, Any] | None:
    """Deferral fields while the pause flag blocks new worker launches, else ``None``.

    Issue #2766: a fleet pass walks every registered repo for minutes, but the
    supervisor reads the flag only between fleet passes, so a pause written
    part-way through one left every repo it had not reached free to launch a
    full cap of workers. ``worker_launch_gate.issue_worker_launch_permit`` calls
    this at its top -- lock-free, before the fleet lock -- so each lane's
    launch decision reads the flag fresh from disk (never a copy cached at
    pass start) and a paused fleet never contends for the lock.

    Only *starting* work is gated: running workers, in-flight reviews,
    reconcile and salvage do not pass through the permit and are untouched.
    ``--ignore-pause`` (``app.ignore_pause``) bypasses it. Existence is the
    signal (:func:`read_fleet_pause`), so an unreadable flag still blocks.
    """
    if app.ignore_pause:
        return None
    flag = read_fleet_pause(app.fleet_dir_override)
    if flag is None:
        return None
    # ``pause_reason``, not ``reason``: ``fleet_dispatch._add_skip_reasons``
    # reads ``data["reason"]`` as a skip reason ahead of ``deferred_reason``.
    return {"paused_at": flag.get("paused_at"), "pause_reason": flag.get("reason")}


def _command_label(args: argparse.Namespace) -> str:
    fleet_command = getattr(args, "fleet_command", None)
    parts = [str(args.command)] + ([str(fleet_command)] if fleet_command else [])
    return " ".join(parts)


def refuse_pass_while_paused(args: argparse.Namespace) -> CommandResult | None:
    """Pass-entry gate: ``None`` to proceed, else the ``fleet_paused`` refusal.

    ``fleet supervise`` reads the flag at its own loop boundary and
    ``fleet-pass.ps1`` before launching; this covers the one-shot entrypoints
    (``fleet work|bash-rats``, ``work``, ``bash-rats``), which previously ran
    ``self_deploy()`` and a full pass into a paused fleet. Runs before any of
    that, from ``cli.main``, so a refused call moves no HEAD and dispatches
    nothing.

    The refusal's ``data["fleet_paused"]`` makes ``main`` exit
    ``EXIT_FLEET_PAUSED``, like the supervisor; ``ok`` is True for the same
    reason (the pause is the operator's own act, not a failure). Commands that
    are not marked pass-running are never gated.
    """
    if not getattr(args, PASS_RUNNING_DEST, False):
        return None
    fleet_dir_override = getattr(args, "fleet_dir", None)
    flag = read_fleet_pause(fleet_dir_override)
    if flag is None:
        return None
    command = _command_label(args)
    flag_path = layout.fleet_pause_path(fleet_dir_override)
    reason = flag.get("reason")
    if ignore_pause_requested(args):
        logger.warning(
            "--ignore-pause: running `charlie %s` although the fleet is paused (%s present, "
            "paused_at=%s, reason=%s)",
            command,
            flag_path,
            flag.get("paused_at"),
            reason,
        )
        return None
    message = (
        f"fleet paused ({layout.FLEET_PAUSE_FILENAME} present at {flag_path}); "
        f"`charlie {command}` not started -- no self-deploy, no dispatch. "
        "`charlie fleet resume` clears the pause; --ignore-pause forces a one-shot."
    )
    logger.warning(message)
    return CommandResult(
        True,
        message,
        {
            "fleet_paused": True,
            "exit_reason": FLEET_PAUSED,
            "command": command,
            "flag": str(flag_path),
            "paused_at": flag.get("paused_at"),
            "reason": reason,
        },
    )


def _supervisor_live(fleet_dir_override: str | None) -> bool:
    heartbeat = read_supervisor_heartbeat(fleet_dir_override)
    return bool(
        heartbeat
        and not heartbeat.get("exited_at")
        and isinstance(heartbeat.get("pid"), int)
        and _host.current().probe.is_alive(heartbeat["pid"])
    )


def register_fleet_pause_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register ``fleet pause`` / ``fleet resume`` (parser wiring lives here
    because cli.py is under the file-size ratchet, #1442)."""
    pause = subparsers.add_parser(
        "pause",
        help=(
            "Pause the fleet (#1776): write fleet-pause.json so the supervisor "
            "exits cleanly at its next pass boundary and the scheduled-task "
            "launcher refuses to restart it. Nothing is killed; live workers "
            "and CI listeners are untouched. Undo with `fleet resume`."
        ),
    )
    pause.add_argument("--reason", default=None, help="Free-text reason recorded in the flag.")
    subparsers.add_parser(
        "resume",
        help="Remove the pause flag; the next watchdog tick relaunches the supervisor (#1776).",
    )


def run_fleet_pause(args: argparse.Namespace) -> CommandResult:
    """Write the pause flag and report what it does and does not stop."""
    flag_path = layout.fleet_pause_path(args.fleet_dir)
    reason = getattr(args, "reason", None)
    if args.dry_run:
        return CommandResult(
            True,
            f"dry-run: would write pause flag to {flag_path}",
            {"dry_run": True, "flag": str(flag_path), "reason": reason},
        )
    already = read_fleet_pause(args.fleet_dir) is not None
    path = write_fleet_pause(args.fleet_dir, reason=reason)
    live = _supervisor_live(args.fleet_dir)
    message = f"fleet paused ({path})"
    if already:
        message += " [was already paused; reason updated]"
    message += (
        "; the supervisor exits at its next pass boundary (workers and listeners untouched)"
        if live
        else "; no live supervisor detected"
    )
    message += f"; {PAUSE_CONSEQUENCE}"
    return CommandResult(
        True,
        message,
        {
            "flag": str(path),
            "reason": reason,
            "already_paused": already,
            "supervisor_live": live,
            "consequence": PAUSE_CONSEQUENCE,
        },
    )


def run_fleet_resume(args: argparse.Namespace) -> CommandResult:
    """Remove the pause flag."""
    flag_path = layout.fleet_pause_path(args.fleet_dir)
    if args.dry_run:
        return CommandResult(
            True,
            f"dry-run: would remove pause flag {flag_path}",
            {"dry_run": True, "flag": str(flag_path)},
        )
    removed = clear_fleet_pause(args.fleet_dir)
    if removed:
        try:
            write_gate = _fleet_write_gate(args.fleet_dir)
            write_gate.log_event(
                # event-consumer: audit-only -- operator resume record; the flag's absence
                # (and ``fleet status``) is the live signal, this row is the audit trail
                kind=FLEET_RESUMED,
                payload={"flag": str(flag_path)},
            )
        except Exception:  # noqa: BLE001 - audit must not mask the removal
            logger.warning("could not record %s event", FLEET_RESUMED, exc_info=True)
    message = (
        "fleet resumed; the next charlie-fleet-pass tick relaunches the supervisor"
        if removed
        else "fleet was not paused"
    )
    return CommandResult(True, message, {"flag": str(flag_path), "removed": removed})
