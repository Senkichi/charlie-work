"""CLI command layer for the recorded-vs-collected gate (issue #1621).

This module is the command wrapper for
:mod:`charlie_work.junit_recorded_gate`, following the same split as
:mod:`charlie_work.collect_only_gate_command` and
:mod:`charlie_work.private_slug_check_command` (subparser registration +
file I/O + exit-code decision here; pure scanning logic in the gate module).

The gate accepts a pre-written junit XML file (``--junit``) and a pre-captured
``pytest --collect-only`` output file (``--collect``).  The CI workflow is
responsible for running ``pytest --collect-only -q`` once and the suite with
``--junit-xml=<file>``, then invoking this command; this keeps the command
testable (the pure logic is tested with string fixtures; the command is tested
with temp files) and avoids duplicating the test-runner logic that belongs in
the CI workflow.

This gate is **enforcement**: it returns ``ok=False`` when the junit
``<testcase>`` count differs from the collected count (or when either side is
emptily zero, or the junit document is internally inconsistent).  In CI it
runs as a step of the already-required "Tests" job, so a failed gate blocks
the merge with no new required-check name and no coordinated three-place
config edit.

``cli`` is imported lazily *inside* the functions that need it, for the same
circular-import / ``-m`` guard reasons documented in
:mod:`charlie_work.private_slug_check_command`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .junit_recorded_gate import (
    JunitRecordedResult,
    compare_recorded_vs_collected,
    render_gate_report,
)
from .workflow import CommandResult


def register_junit_recorded_check_subparser(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``junit-recorded-check`` subcommand on *subparsers*."""
    parser = subparsers.add_parser(
        "junit-recorded-check",
        help=(
            "CI gate (issue #1621): fail when the junit <testcase> count "
            "differs from the pytest --collect-only count. Under load a "
            "full pytest --junit-xml run can exit 0 with zero failures yet "
            "write a junit file that silently omits a contiguous tail of "
            "tests/; this gate makes that shape a merge failure. Runs as a "
            "step of the required 'Tests' job, so no new required-check "
            "name is introduced."
        ),
    )
    parser.add_argument(
        "--junit",
        required=True,
        help=(
            "Path to the junit XML file written by "
            "'pytest --junit-xml=<file>'. May be relative to the repo root."
        ),
    )
    parser.add_argument(
        "--collect",
        required=True,
        help=(
            "Path to a file containing the stdout of "
            "'pytest --collect-only -q' run against the same tree. The "
            "gate sums the per-file counts (pytest 9.x -q format) or reads "
            "the 'N tests collected' summary line (non-q format)."
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


def _read_file(path: Path, label: str) -> str:
    """Read a gate input file, raising ``ConfigError`` on failure.

    Fail closed (matching the collect-only / private-slug gates): a missing
    or unreadable input is a gate failure, not a silent pass.
    """
    from .config import ConfigError

    if not path.exists():
        raise ConfigError(
            f"junit-recorded-check: {label} file not found: {path}. "
            f"Ensure the CI workflow ran the corresponding pytest command "
            f"and wrote the output to this path."
        )
    try:
        return path.read_text(encoding="utf-8", errors="surrogateescape")
    except OSError as exc:
        raise ConfigError(
            f"junit-recorded-check: could not read {label} file {path}: {exc}"
        ) from exc


def run_junit_recorded_check_command(
    args: argparse.Namespace,
) -> CommandResult:
    """CI gate (issue #1621): fail when recorded != collected test count.

    Reads the junit XML file (``--junit``) and the collect-only output file
    (``--collect``), runs the pure comparison logic
    (:func:`compare_recorded_vs_collected`), and returns ``ok=False`` when
    the junit ``<testcase>`` count differs from the collected count, when
    either side is emptily zero, or when the junit document is internally
    inconsistent.

    Errors as values (per CLAUDE.md): file I/O failures come back as
    ``CommandResult(ok=False)`` -- never raised -- so the CI step exits
    non-zero without a Python traceback.
    """
    from . import cli  # deferred: see module docstring (circular-import / -m guard)

    ctx = cli.bootstrap_command(args)
    junit_path = ctx.repo_root / getattr(args, "junit")
    collect_path = ctx.repo_root / getattr(args, "collect")

    try:
        junit_xml = _read_file(junit_path, "junit")
    except Exception as exc:
        return CommandResult(
            False,
            f"junit-recorded-check: {exc}",
            {"junit": str(junit_path)},
        )
    try:
        collect_output = _read_file(collect_path, "collect")
    except Exception as exc:
        return CommandResult(
            False,
            f"junit-recorded-check: {exc}",
            {"collect": str(collect_path)},
        )

    result: JunitRecordedResult = compare_recorded_vs_collected(junit_xml, collect_output)
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

    data: dict[str, Any] = {
        "junit": str(junit_path),
        "collect": str(collect_path),
        "recorded": result.recorded,
        "collected": result.collected,
        "suite_tests_attr": result.suite_tests_attr,
        "findings": [{"kind": f.kind, "detail": f.detail} for f in result.findings],
    }

    if result.ok:
        return CommandResult(
            True,
            f"junit-recorded-check: PASSED ({result.recorded} recorded == "
            f"{result.collected} collected)",
            data,
        )

    finding_lines = [f"  {f.kind}: {f.detail}" for f in result.findings]
    message = (
        f"junit-recorded-check: FAILED ({len(result.findings)} finding(s))\n"
        + "\n".join(finding_lines)
        + f"\nRecorded: {result.recorded}, Collected: {result.collected}, "
        f"suite tests attr sum: {result.suite_tests_attr}.\n"
        f"A junit file that records fewer tests than were collected fails "
        f"the merge even when pytest exits 0 with zero failures -- the "
        f"truncated-tail failure mode of issue #1621."
    )
    return CommandResult(False, message, data)
