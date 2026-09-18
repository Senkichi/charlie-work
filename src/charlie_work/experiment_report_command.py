"""CLI command layer for the per-PR experiment read-out (issue #1701).

``charlie experiment-report --experiment review_effort`` is the
operator-facing entry point for the reviewer-effort experiment that has
been recording arm assignments in ``events.db`` since 2026-07-26 without a
consumer.  Workers cannot reach the live ``events.db``, so the deliverable
is a read-only command the operator runs against runtime state.

Read-only contract (asserted by the acceptance tests):

* ``instrumentation._get_db`` mutates on first open -- WAL pragmas, schema
  creation/migration, and the legacy ``events.jsonl`` import.  The command
  therefore gates on ``events.db`` existence *before* calling
  :func:`charlie_work.instrumentation.query_events`, and refuses outright
  when an unmigrated ``events.jsonl`` sits next to the database (reading
  then would trigger the auto-migration -- a write).  With a normal WAL
  database and no stray ``events.jsonl``, ``query_events`` is pure SELECT
  on an already-migrated schema, so the command performs no writes at all:
  ``state.json`` and ``events.db`` come out byte-identical.
* ``--experiment`` selects the session-metrics key that carries the arm:
  ``review_effort`` -> ``review_effort_arm``; a value already ending in
  ``_arm`` is used verbatim.  The implementation never names an arm value;
  arms are derived from whatever values the key recorded.

``cli`` is imported lazily inside the run function for the same
circular-import / ``-m`` guard reasons documented in
:mod:`charlie_work.private_slug_check_command`.
"""

from __future__ import annotations

import argparse
from typing import Any

from .experiment_report import (
    DEFAULT_MIN_PRS_PER_ARM,
    build_report,
    metrics_key_for_experiment,
    parse_window_bound,
    render_text,
)
from .workflow import CommandResult


def register_experiment_report_subparser(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``experiment-report`` subcommand on *subparsers*."""
    parser = subparsers.add_parser(
        "experiment-report",
        help=(
            "Read-only per-PR read-out of a reviewer experiment from "
            "events.db (issue #1701). Groups every metric by PR -- the arm "
            "assignment is stable per PR across rework rounds -- labels each "
            "metric activity or outcome, and evaluates the experiment's "
            "documented stopping rule. Writes nothing."
        ),
        description=(
            "Read-only per-PR read-out of a reviewer experiment from "
            "events.db (issue #1701). The report aggregates by PR (the unit "
            "of randomization), separates activity metrics from outcome "
            "metrics, gives each rate a 95% interval plus pairwise "
            "difference intervals, and evaluates the stopping rule "
            "documented in docs/review-effort-experiment.md."
        ),
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help=(
            "Experiment name; selects the session-metrics key carrying the "
            "arm. '<name>' reads '<name>_arm' (e.g. review_effort -> "
            "review_effort_arm); a value already ending in '_arm' is used "
            "verbatim."
        ),
    )
    parser.add_argument(
        "--since",
        default=None,
        metavar="ISO-8601",
        help=(
            "Only events at or after this timestamp contribute to any "
            "figure (inclusive; naive values are read as UTC)."
        ),
    )
    parser.add_argument(
        "--until",
        default=None,
        metavar="ISO-8601",
        help=(
            "Only events at or before this timestamp contribute to any "
            "figure (inclusive; naive values are read as UTC)."
        ),
    )
    parser.add_argument(
        "--exclude-window",
        nargs=2,
        action="append",
        metavar=("START", "END"),
        default=None,
        help=(
            "Inclusive ISO-8601 [START, END] window whose events contribute "
            "to no figure; repeatable. Events one second outside a window "
            "still count."
        ),
    )
    parser.add_argument(
        "--min-prs-per-arm",
        type=int,
        default=DEFAULT_MIN_PRS_PER_ARM,
        help=(
            f"Stopping-rule minimum assigned PRs per arm (default: "
            f"{DEFAULT_MIN_PRS_PER_ARM}; the equivalence bound is twice "
            "this). Analysis knob only -- the documented rule is fixed at "
            "the default."
        ),
    )


def _parse_windows(args: argparse.Namespace) -> tuple[Any, Any, list[tuple[Any, Any]]]:
    """Parse ``--since``/``--until``/``--exclude-window`` into datetimes.

    Raises ``ValueError`` naming the offending flag; the caller turns that
    into a ``CommandResult(False, ...)`` (errors as values).
    """
    since = parse_window_bound(getattr(args, "since", None))
    until = parse_window_bound(getattr(args, "until", None))
    if since is not None and until is not None and since > until:
        raise ValueError(f"--since {args.since!r} is after --until {args.until!r}")
    exclude: list[tuple[Any, Any]] = []
    for pair in getattr(args, "exclude_window", None) or []:
        start = parse_window_bound(pair[0])
        end = parse_window_bound(pair[1])
        if start > end:
            raise ValueError(f"--exclude-window start {pair[0]!r} is after end {pair[1]!r}")
        exclude.append((start, end))
    return since, until, exclude


def run_experiment_report_command(args: argparse.Namespace) -> CommandResult:
    """Read-only per-PR experiment read-out from ``events.db``.

    Fetches the full event log through ``instrumentation.query_events``
    (window filtering lives in :func:`build_report` so ``--exclude-window``
    has one inclusive semantics), builds the report, and returns the text
    rendering as ``message``; under ``--json`` the structured report is the
    ``data`` payload.
    """
    from . import cli  # deferred: see module docstring
    from . import instrumentation

    ctx = cli.bootstrap_command(args)
    state_path = ctx.paths.state_file
    metrics_key = metrics_key_for_experiment(getattr(args, "experiment"))

    # Read-only gate: _get_db would create/migrate the database on open, so
    # a missing database must never reach query_events. An unmigrated
    # events.jsonl would trigger the legacy import -- also a write -- so the
    # command refuses rather than mutate the log it is meant only to read.
    db_path = instrumentation._db_path(state_path)
    if not db_path.exists():
        return CommandResult(
            False,
            f"experiment-report: events.db not found at {db_path} -- "
            "no instrumentation events have been recorded for this repo yet.",
            {},
        )
    jsonl_path = instrumentation._jsonl_path(state_path)
    if jsonl_path.exists():
        return CommandResult(
            False,
            f"experiment-report: refusing to read: {jsonl_path} exists "
            "unmigrated and query_events would auto-migrate it (a write). "
            "Run any read-write charlie command once to migrate, then "
            "re-run this report.",
            {},
        )

    try:
        since, until, exclude = _parse_windows(args)
    except ValueError as exc:
        return CommandResult(False, f"experiment-report: {exc}", {})

    min_prs = getattr(args, "min_prs_per_arm", DEFAULT_MIN_PRS_PER_ARM)
    if not isinstance(min_prs, int) or min_prs < 1:
        return CommandResult(
            False,
            f"experiment-report: --min-prs-per-arm must be a positive integer, got {min_prs!r}",
            {},
        )

    events = instrumentation.query_events(state_path)
    report = build_report(
        events,
        metrics_key,
        since=since,
        until=until,
        exclude_windows=exclude,
        min_prs_per_arm=min_prs,
    )
    report["events_db"] = str(db_path)

    if getattr(args, "json_output", False):
        return CommandResult(
            True,
            f"experiment-report: {metrics_key} ({report['events_scanned']} events)",
            report,
        )
    return CommandResult(True, render_text(report), {})
