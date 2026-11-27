"""``charlie gang`` subcommands (issue #3117, ADR-0010): create and inspect foremen.

``gang`` is the operator's front door to the foreman layer: ``new`` scaffolds
``foremen/<name>/``, ``list`` shows every foreman with its state and bounds
usage, and ``show`` prints one foreman's charter, cached state, and open
items.

This module is the *only* seam into ``charlie_work.foreman``: every import of
that package happens inside the handlers below, so importing the CLI never
pulls the foreman layer in and nothing else can reach it. The import-graph
guard lives in ``tests/test_foreman_import_direction.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

from .command_result import CommandResult
from .fleet_paths import fleet_dir

GANG_COMMAND = "gang"


def register_gang_subparsers(subparsers: Any) -> None:
    """Add ``gang {new,list,show}`` to the top-level subparsers."""
    gang = subparsers.add_parser(
        GANG_COMMAND, help="Manage foremen — chartered outcome managers (ADR-0010)"
    )
    sub = gang.add_subparsers(dest="gang_command", required=True)
    new = sub.add_parser("new", help="Create a foreman under foremen/<name>/")
    new.add_argument("name", help="foreman name ([a-z0-9][a-z0-9-]*)")
    new.add_argument(
        "--from",
        dest="from_dir",
        type=str,
        default=None,
        metavar="DIR",
        help="copy charter.yaml and instructions.md from DIR instead of the built-in minimum",
    )
    sub.add_parser("list", help="List foremen with state and bounds usage")
    show = sub.add_parser("show", help="Print one foreman's charter, state, and open items")
    show.add_argument("name", help="foreman name")


def _plain(value: Any) -> Any:
    """Dataclass tree -> JSON-ready dict."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _gang_new(args: argparse.Namespace, store: Any, fleet: Any) -> CommandResult:
    from_dir = Path(args.from_dir) if args.from_dir else None
    created = store.scaffold_foreman(fleet, args.name, from_dir=from_dir)
    if not created.ok:
        return CommandResult(False, f"gang new: {created.error}", {})
    return CommandResult(
        True,
        f"created foreman {args.name!r} at {created.value}",
        {"name": args.name, "dir": str(created.value)},
    )


def _gang_list(
    fcharter: Any, fitems: Any, fstate: Any, store: Any, display_state: Any, fleet: Any
) -> CommandResult:
    foremen = store.list_foremen(fleet)
    if not foremen:
        return CommandResult(True, "no foremen", {"foremen": []})
    repos = store.registered_repo_keys(fleet)
    lines: list[str] = []
    data: list[dict[str, Any]] = []
    for name, fdir in foremen:
        state = fstate.read_state_snapshot(fdir)
        open_count = len(fitems.open_items(fitems.load_items(fdir)))
        entry: dict[str, Any] = {
            "name": name,
            "state": state.state,
            "display_state": display_state(state.state),
            "open_items": open_count,
        }
        chartered = fcharter.load_charter(fdir, registered_repos=repos)
        if chartered.ok:
            bounds = chartered.value.bounds
            entry["issues"] = f"{len(state.issues)}/{bounds.max_open_issues}"
            entry["turns_today"] = f"{state.turns_today}/{bounds.max_turns_per_day}"
            lines.append(
                f"{name}: {display_state(state.state)}, issues {entry['issues']}, "
                f"turns today {entry['turns_today']}, items {open_count} open"
            )
        else:
            entry["charter_error"] = chartered.error
            lines.append(
                f"{name}: {display_state(state.state)}, items {open_count} open "
                f"(charter invalid: {chartered.error})"
            )
        data.append(entry)
    return CommandResult(True, "\n".join(lines), {"foremen": data})


def _gang_show(
    fcharter: Any, fitems: Any, fstate: Any, store: Any, display_state: Any, fleet: Any, name: str
) -> CommandResult:
    resolved = store.resolve_foreman(fleet, name)
    if not resolved.ok:
        return CommandResult(False, f"gang show: {resolved.error}", {})
    fdir = resolved.value
    state = fstate.read_state_snapshot(fdir)
    chartered = fcharter.load_charter(fdir, registered_repos=store.registered_repo_keys(fleet))
    open_list = fitems.open_items(fitems.load_items(fdir))
    data: dict[str, Any] = {
        "name": name,
        "dir": str(fdir),
        "state": _plain(state),
        "display_state": display_state(state.state),
        "open_items": [_plain(item) for item in open_list],
    }
    lines = [f"{name}: {display_state(state.state)}", f"  dir: {fdir}"]
    if chartered.ok:
        charter = chartered.value
        data["charter"] = _plain(charter)
        lines += [
            f"  outcome: {charter.outcome}",
            f"  repos: {list(charter.repos)}",
            f"  standing: {charter.standing}",
            f"  arming: {charter.arming}",
            f"  bounds: issues {len(state.issues)}/{charter.bounds.max_open_issues}, "
            f"per turn {charter.bounds.max_issues_per_turn}, "
            f"turns today {state.turns_today}/{charter.bounds.max_turns_per_day}, "
            f"min interval {charter.bounds.min_turn_interval_minutes}m, "
            f"quiet {charter.bounds.max_quiet_hours}h, "
            f"unanswered {charter.bounds.max_unanswered_days}d",
        ]
        if charter.done_when is not None:
            lines.append(f"  done_when.check: {list(charter.done_when.check)}")
        if charter.cadence is not None and charter.cadence.report_at:
            lines.append(f"  cadence.report_at: {charter.cadence.report_at}")
    else:
        data["charter_error"] = chartered.error
        lines.append(f"  charter invalid: {chartered.error}")
    if open_list:
        lines.append("  open items:")
        for item in open_list:
            label = item.question or item.text or ""
            lines.append(f"    {item.id} [{item.kind}] {label}")
    else:
        lines.append("  open items: none")
    return CommandResult(True, "\n".join(lines), data)


def run_gang_command(args: argparse.Namespace) -> CommandResult:
    """Dispatch ``charlie gang <sub>``; all failures come back as values."""
    # The one sanctioned edge into charlie_work.foreman: a function-local
    # import so `import charlie_work.cli` never loads the foreman package.
    from .foreman import charter as fcharter
    from .foreman import display_state
    from .foreman import items as fitems
    from .foreman import state as fstate
    from .foreman import store

    fleet = fleet_dir(override=args.fleet_dir)
    if args.gang_command == "new":
        return _gang_new(args, store, fleet)
    if args.gang_command == "list":
        return _gang_list(fcharter, fitems, fstate, store, display_state, fleet)
    if args.gang_command == "show":
        return _gang_show(fcharter, fitems, fstate, store, display_state, fleet, args.name)
    return CommandResult(False, f"unknown gang command: {args.gang_command}", {})
