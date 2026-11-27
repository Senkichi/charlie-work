"""``charlie scheme`` (issue #3122, ADR-0010): run one foreman pass.

The operator's way to run the pass on demand — the fleet supervise loop
spawns the same argv on ``foreman.pass_interval_minutes``. One pass iterates
every foreman: wake predicate, deferral bounds, and a Turn (Collect → Decide
→ Apply) where the predicate holds. It exits when the iteration ends.

This module is the second sanctioned seam into ``charlie_work.foreman``
(the other is ``gang_command``): the package import happens inside
``run_scheme_command`` so importing the CLI never loads the foreman layer.
The import-graph guard lives in ``tests/test_foreman_import_direction.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Any

from .command_result import CommandResult
from .fleet_paths import fleet_dir

SCHEME_COMMAND = "scheme"


def register_scheme_subparsers(subparsers: Any) -> None:
    """Add the ``scheme`` parser to the top-level subparsers."""
    # Deferred (the sibling *_command pattern): ``cli`` owns the global-aware
    # ``--dry-run`` idiom — a plain store_true subcommand flag overwrites the
    # top-level one argparse already parsed. A module-level edge would re-enter
    # ``cli`` while it is still initialising: it imports this module to wire the
    # subparser.
    from .cli import _add_dry_run

    scheme = subparsers.add_parser(
        SCHEME_COMMAND, help="Run one foreman pass over the fleet's foremen (ADR-0010)"
    )
    scheme.add_argument(
        "--only",
        metavar="NAME",
        default=None,
        help="restrict the pass to one foreman",
    )
    _add_dry_run(
        scheme,
        help="run Collect and Decide and print what Apply would do — no effect",
    )
    scheme.add_argument(
        "--no-decide",
        action="store_true",
        dest="no_decide",
        help="with --dry-run: stop after writing briefing.md (launches no model)",
    )


def _format_outcome(outcome: Any) -> str:
    name, kind = outcome.name, outcome.outcome
    if kind == "turn":
        return f"{name}: Turn {outcome.turn_id}: {outcome.detail}"
    if kind == "idle":
        return f"{name}: idle"
    if kind == "deferred":
        suffix = f" — {outcome.detail}" if outcome.detail else ""
        return f"{name}: deferred ({outcome.reason}){suffix}"
    if kind == "skipped":
        return f"{name}: skipped ({outcome.reason})"
    if kind == "failed":
        return f"{name}: failed — {outcome.detail}"
    if kind == "dry_run":
        if outcome.reason == "no_decide":
            return f"{name}: dry-run — {outcome.detail}"
        return f"{name}: dry-run Turn {outcome.turn_id}:\n{_format_plan(outcome.detail)}"
    return f"{name}: {kind}"


def _format_plan(detail: str | None) -> str:
    """Render the dry-run Apply plan from its JSON detail."""
    try:
        plan = json.loads(detail or "{}")
    except ValueError:
        return f"  {detail}"
    lines = [f"  {plan.get('turn_summary', '')}"]
    for action in plan.get("plan", []):
        reason = f" — {action['reason']}" if action.get("reason") else ""
        key = f"[{action['key']}]" if action.get("key") else ""
        lines.append(f"  {action.get('type')}{key}: {action.get('outcome')}{reason}")
    return "\n".join(lines)


def run_scheme_command(args: argparse.Namespace) -> CommandResult:
    """Dispatch ``charlie scheme``; all failures come back as values."""
    # Sanctioned edge into charlie_work.foreman (with gang_command.py):
    # function-local so `import charlie_work.cli` never loads the package.
    from .foreman import pass_run

    fleet = fleet_dir(override=args.fleet_dir)
    result = pass_run.run_pass(
        fleet,
        only=args.only,
        dry_run=args.dry_run,
        no_decide=args.no_decide,
    )
    if not result.ok:
        return CommandResult(False, f"scheme: {result.error}", {})
    passed = result.value
    if passed.locked:
        return CommandResult(
            True,
            "foreman pass already running (foremen/pass.lock held)",
            {"locked": True},
        )
    if passed.disabled:
        return CommandResult(
            True, "foreman layer disabled (foreman.enabled=false)", {"disabled": True}
        )
    if not passed.outcomes:
        return CommandResult(True, "no foremen", {"outcomes": []})
    failed = any(outcome.outcome == "failed" for outcome in passed.outcomes)
    return CommandResult(
        not failed,
        "\n".join(_format_outcome(outcome) for outcome in passed.outcomes),
        {"outcomes": [dataclasses.asdict(outcome) for outcome in passed.outcomes]},
    )
