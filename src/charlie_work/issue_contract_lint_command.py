"""CLI command layer for the issue-body contract lint (issue #3115).

``charlie lint-issue --body <file> --blockers "<csv>"`` lints one or more
issue-body files against the dispatchable-body contract in
:mod:`charlie_work.issue_contract_lint`, printing one JSON document with the
repo's Ready label (from the layered config) and a result per body. The flags
repeat as positional pairs: the Nth ``--blockers`` is the expected blocker set
for the Nth ``--body``.

Exit codes: 0 when every body is clean, 1 when any body fails, 2 when the
repo's fleet config cannot be loaded (handled by ``cli.main``'s standard
``ConfigError``/``ValueError`` mapping — the command raises, never exits).

Subparser registration + file I/O + the exit-code decision live here; the
rules themselves are pure functions in ``issue_contract_lint``. ``cli`` is
imported lazily inside the run function for the same circular-import / ``-m``
guard reasons documented in :mod:`charlie_work.private_slug_check_command`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .command_result import CommandResult
from .config import ConfigError
from .github_body_scan import parse_blockers
from .issue_contract_lint import lint_issue_body


def register_lint_issue_subparser(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``lint-issue`` subcommand on *subparsers*."""
    parser = subparsers.add_parser(
        "lint-issue",
        help=(
            "Lint issue-body files against the dispatchable-body contract "
            "(issue #3115): declared blockers must equal the expected set, no "
            "prose-only dependencies, no path:line citations, no "
            "closing-keyword issue references. Prints one JSON document with "
            "the repo's Ready label and a result per body. Exit 0 when all "
            "pass, 1 when a body fails, 2 when the repo's fleet config cannot "
            "be loaded."
        ),
    )
    parser.add_argument(
        "--body",
        action="append",
        required=True,
        type=Path,
        metavar="FILE",
        help=(
            "Issue-body file to lint; repeat to lint several bodies. The Nth "
            "--body pairs with the Nth --blockers. Relative paths resolve "
            "against the current directory."
        ),
    )
    parser.add_argument(
        "--blockers",
        action="append",
        required=True,
        metavar="CSV",
        help=(
            "Comma-separated issue numbers the paired --body is expected to "
            "declare as blockers; pass '' for none. Repeatable, paired "
            "positionally with --body."
        ),
    )


def _parse_blocker_csv(spec: str) -> frozenset[int]:
    """Parse a ``--blockers`` CSV value into a frozenset of issue numbers."""
    numbers: set[int] = set()
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            numbers.add(int(piece))
        except ValueError:
            raise ConfigError(
                f"lint-issue: --blockers value {spec!r} is not a comma-separated "
                f"list of issue numbers (bad item {piece!r})"
            ) from None
    return frozenset(numbers)


def run_lint_issue_command(args: argparse.Namespace) -> CommandResult:
    """Lint each ``--body`` file; ``CommandResult.data`` is the JSON document.

    The document is ``{"ok", "ready_label", "results"}`` where each result is
    ``{"body", "ok", "expected_blockers", "blockers", "problems"}`` — a body
    that cannot be read fails its own entry rather than aborting the run, so
    the document always carries one result per body.
    """
    from . import cli  # deferred: see module docstring (circular-import / -m guard)

    bodies: list[Path] = list(args.body)
    blocker_specs: list[str] = list(args.blockers)
    if len(bodies) != len(blocker_specs):
        raise ConfigError(
            f"lint-issue: --body and --blockers must be given in pairs "
            f"({len(bodies)} body file(s), {len(blocker_specs)} blocker spec(s))"
        )
    expected = [_parse_blocker_csv(spec) for spec in blocker_specs]

    # Read-only command: resolve the repo root without redirecting a linked
    # worktree to the main checkout (issue #1600 precedent), so relative
    # config layers and --body paths bind to the tree the caller is in.
    ctx = cli.bootstrap_command(args, redirect_to_main_worktree=False)

    results: list[dict[str, Any]] = []
    all_ok = True
    for body_path, expected_set in zip(bodies, expected):
        result: dict[str, Any] = {
            "body": str(body_path),
            "expected_blockers": sorted(expected_set),
        }
        try:
            text = body_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            result["ok"] = False
            result["blockers"] = None
            result["problems"] = [f"cannot read body file: {exc}"]
            all_ok = False
        else:
            problems = lint_issue_body(text, expected_set)
            result["ok"] = not problems
            result["blockers"] = parse_blockers(text)
            result["problems"] = list(problems)
            if problems:
                all_ok = False
        results.append(result)

    failed = sum(1 for r in results if not r["ok"])
    document = {
        "ok": all_ok,
        "ready_label": ctx.config.labels.ready,
        "results": results,
    }
    return CommandResult(
        all_ok,
        f"lint-issue: {len(results)} body file(s), {failed} failed",
        document,
    )
