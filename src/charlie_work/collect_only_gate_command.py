"""CLI command layer for the collect-only gate (issue #1538).

This module is the command wrapper for
:mod:`charlie_work.collect_only_gate`, following the same split as
:mod:`charlie_work.private_slug_check_command` and
:mod:`charlie_work.ast_equivalence_gate_command` (subparser registration +
file I/O + exit-code decision here; pure scanning logic in the gate module).

The gate accepts pre-collected ``pytest --collect-only -q`` output via
``--base-collect`` and ``--head-collect`` file paths.  The CI workflow is
responsible for running ``pytest --collect-only -q`` at both the base and head
refs (creating a git worktree at the base ref, syncing deps, and capturing
output); this command only reads the two output files and runs the pure
comparison logic.  This keeps the command testable (the pure logic is tested
with string fixtures; the command is tested with temp files) and avoids
duplicating the worktree/venv-management logic that belongs in the CI workflow.

Unlike the AST-equivalence gate (#1541, evidence only -- always ``ok=True``),
this gate is **enforcement**: it returns ``ok=False`` when the leaf-name
multisets differ or a removed leaf did not reappear in a sibling.  The CI job
is a required check, so a failed gate blocks the merge.

``cli`` is imported lazily *inside* the functions that need it, for the same
circular-import / ``-m`` guard reasons documented in
:mod:`charlie_work.private_slug_check_command`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .collect_only_gate import CollectOnlyResult, compare_collect_only, render_gate_report
from .workflow import CommandResult


def register_collect_only_check_subparser(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``collect-only-check`` subcommand on *subparsers*."""
    parser = subparsers.add_parser(
        "collect-only-check",
        help=(
            "CI gate (issue #1538): compare leaf-name multisets from "
            "pytest --collect-only -q output at base and head. A verbatim "
            "test relocation (same leaf name, different module path) passes; "
            "a rename, addition, deletion, or class-wrapping dodge does not. "
            "Additionally asserts every leaf removed from a source module "
            "under tests/ reappears in a sibling under tests/. This gate is "
            "enforcement (a required check), not evidence."
        ),
    )
    parser.add_argument(
        "--base-collect",
        required=True,
        help=(
            "Path to a file containing the stdout of "
            "'pytest --collect-only' run against the base ref. The CI "
            "workflow creates this by checking out the base ref (via a git "
            "worktree), syncing deps, and running pytest --collect-only. "
            "NOTE: --collect-only WITHOUT -q; pytest 9.x's -q produces a "
            "compact 'file: count' format without individual node IDs."
        ),
    )
    parser.add_argument(
        "--head-collect",
        required=True,
        help=(
            "Path to a file containing the stdout of "
            "'pytest --collect-only' run against the head ref (the PR's "
            "own checkout)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Write the gate report to this file (in addition to stdout). "
            "When set, also writes to $GITHUB_STEP_SUMMARY if that env var "
            "is set (CI step-summary rendering)."
        ),
    )


def _read_collect_file(path: Path) -> str:
    """Read a collect-only output file, raising ``ConfigError`` on failure.

    Fail closed (matching the private-slug gate's baseline philosophy): a
    missing or unreadable collect file is a gate failure, not a silent pass.
    """
    from .config import ConfigError

    if not path.exists():
        raise ConfigError(
            f"collect-only-check: collect file not found: {path}. "
            f"Ensure the CI workflow ran 'pytest --collect-only -q' and "
            f"wrote the output to this path."
        )
    try:
        return path.read_text(encoding="utf-8", errors="surrogateescape")
    except OSError as exc:
        raise ConfigError(
            f"collect-only-check: could not read collect file {path}: {exc}"
        ) from exc


def run_collect_only_check_command(
    args: argparse.Namespace,
) -> CommandResult:
    """CI gate (issue #1538): compare leaf-name multisets from base and head.

    Reads the two pre-collected ``pytest --collect-only -q`` output files
    (``--base-collect`` and ``--head-collect``), runs the pure comparison
    logic (:func:`compare_collect_only`), and returns ``ok=False`` when the
    multisets differ or a removed leaf did not reappear in a sibling.

    Unlike the AST-equivalence gate (#1541, always ``ok=True``), this gate is
    **enforcement**: a failed gate blocks the merge (the CI job is a required
    check).  The gate's interface supports the positive control exercised in
    #1542 (a deliberately wrong split must make the diff non-empty) -- it fails
    loudly on a genuinely missing leaf, not just on a renamed one.

    Errors as values (per CLAUDE.md): file I/O failures come back as
    ``CommandResult(ok=False)`` -- never raised -- so the CI step exits
    non-zero without a Python traceback.
    """
    from . import cli  # deferred: see module docstring (circular-import / -m guard)

    ctx = cli.bootstrap_command(args)
    base_collect_path = ctx.repo_root / getattr(args, "base_collect")
    head_collect_path = ctx.repo_root / getattr(args, "head_collect")

    try:
        base_output = _read_collect_file(base_collect_path)
    except Exception as exc:
        return CommandResult(
            False,
            f"collect-only-check: {exc}",
            {"base_collect": str(base_collect_path)},
        )
    try:
        head_output = _read_collect_file(head_collect_path)
    except Exception as exc:
        return CommandResult(
            False,
            f"collect-only-check: {exc}",
            {"head_collect": str(head_collect_path)},
        )

    result: CollectOnlyResult = compare_collect_only(base_output, head_output)
    report = render_gate_report(result)

    # Write to --output file and/or $GITHUB_STEP_SUMMARY (CI rendering).
    output_path = getattr(args, "output", None)
    if output_path:
        out = ctx.repo_root / output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        try:
            with open(step_summary, "a", encoding="utf-8") as f:
                f.write(report + "\n")
        except OSError:
            pass  # non-fatal: report is also on stdout

    base_total = sum(result.base_leaf_counts.values())
    head_total = sum(result.head_leaf_counts.values())
    data: dict[str, Any] = {
        "base_collect": str(base_collect_path),
        "head_collect": str(head_collect_path),
        "base_leaf_count": base_total,
        "head_leaf_count": head_total,
        "findings": [
            {
                "kind": f.kind,
                "leaf_name": f.leaf_name,
                "source_module": f.source_module,
                "detail": f.detail,
            }
            for f in result.findings
        ],
    }

    if result.ok:
        return CommandResult(
            True,
            f"collect-only-check: PASSED ({base_total} leaf names at base, "
            f"{head_total} at head; multisets match)",
            data,
        )

    finding_lines = [
        f"  {f.kind}: {f.leaf_name}" + (f" (from {f.source_module})" if f.source_module else "")
        for f in result.findings
    ]
    message = (
        f"collect-only-check: FAILED ({len(result.findings)} finding(s))\n"
        + "\n".join(finding_lines)
        + f"\nBase: {base_total} leaf names, Head: {head_total} leaf names.\n"
        f"Multiset mismatch or missing sibling reappearance detected. "
        f"A verbatim test relocation (same leaf name, different module path) "
        f"should pass; a rename, addition, deletion, or class-wrapping dodge "
        f"should fail. See the gate report for details."
    )
    return CommandResult(False, message, data)
