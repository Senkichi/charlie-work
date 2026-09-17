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
this gate is **enforcement**: it returns ``ok=False`` when any finding fails
the issue's Scope table (``removed``, ``missing_sibling``, or
``count_mismatch`` with head < base); ``added`` and ``count_mismatch`` with
head > base are reported but pass, so a pure-addition PR -- the normal
feature/bugfix shape the test-adequacy gate requires -- does not fail.  The
CI job is a required check, so a failed gate blocks the merge.

``cli`` is imported lazily *inside* the functions that need it, for the same
circular-import / ``-m`` guard reasons documented in
:mod:`charlie_work.private_slug_check_command`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .collect_only_gate import (
    EXEMPTION_LOG_MARKER,
    CollectGateExemption,
    CollectOnlyResult,
    compare_collect_only,
    exemption_log_marker,
    render_gate_report,
    resolve_collect_gate_exemption,
)
from .github import label_names
from .workflow import CommandResult


def register_collect_only_check_subparser(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``collect-only-check`` subcommand on *subparsers*."""
    parser = subparsers.add_parser(
        "collect-only-check",
        help=(
            "CI gate (issue #1538): compare leaf-name multisets from "
            "pytest --collect-only output at base and head. A verbatim "
            "test relocation (same leaf name, different module path) passes; "
            "net-new tests (added leaves) and grown multiplicities are "
            "reported but pass; a rename, deletion, shrunken multiplicity, "
            "or class-wrapping dodge fails. Additionally asserts every leaf "
            "removed from a source module under tests/ reappears in a "
            "sibling under tests/. This gate is enforcement (a required "
            "check), not evidence."
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
    parser.add_argument(
        "--pr",
        type=int,
        default=None,
        help=(
            "PR number for the operator exemption (issue #1686). When set, "
            "the command queries the PR's LIVE labels via the GitHub API "
            "and, if the configured collect_gate_exempt label is present, "
            "waives this head's failing findings (exit 0 with every waived "
            "finding still printed). A failed or empty labels query fails "
            "closed -- never read as an exemption. When unset, no exemption "
            "is evaluated."
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
            f"Ensure the CI workflow ran 'pytest --collect-only' and "
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

    Reads the two pre-collected ``pytest --collect-only`` output files
    (``--base-collect`` and ``--head-collect``), runs the pure comparison
    logic (:func:`compare_collect_only`), and returns ``ok=False`` when any
    finding fails the issue's Scope table: ``removed``, ``missing_sibling``,
    or ``count_mismatch`` with head < base.  ``added`` and
    ``count_mismatch`` with head > base are reported but do not fail, so a
    pure-addition PR passes.

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

    # Issue #1686: resolve the operator exemption against the PR's LIVE
    # labels. The query goes through ``GitHub.pr_view`` with a deliberately
    # narrow field set (``labels,headRefOid`` -- same narrowing discipline
    # as the closing-keyword gate's CLOSING_KEYWORD_PR_FIELDS, which exists
    # because the default Actions GITHUB_TOKEN cannot read every nested
    # GraphQL field PR_VIEW_FIELDS pulls). A falsy response means the query
    # failed and resolves to NOT exempt (fail closed); the exemption can
    # only ever be granted on the query's positive answer -- nothing under
    # the PR author's control (PR body, title, commit messages, tree files,
    # or the github.event labels snapshot) participates in this decision.
    exemption: CollectGateExemption | None = None
    head_sha: str | None = None
    pr_number = getattr(args, "pr", None)
    if pr_number is not None:
        # ``headRefOid`` rides the same query: the log marker stamps it so
        # the review packet can prove the evidence describes ITS reviewed
        # head, not an earlier head the label survived (issue #1686).
        pr: dict[str, Any] = {}
        query_error: str | None = None
        try:
            pr = ctx.gh.pr_view(pr_number, fields="labels,headRefOid")
        except Exception as exc:  # GitHubError -- fail closed, never grant
            query_error = f"{type(exc).__name__}: {exc}"
        pr_labels = label_names(pr) if isinstance(pr, dict) and pr else None
        raw_head = pr.get("headRefOid") if isinstance(pr, dict) else None
        head_sha = str(raw_head) if raw_head else None
        if pr_labels is None and query_error is None:
            query_error = "could not fetch PR labels"
        exemption = resolve_collect_gate_exemption(
            exemption_label=ctx.config.labels.collect_gate_exempt,
            pr_number=pr_number,
            pr_labels=pr_labels,
            query_error=query_error,
        )
    waived = result.failures if (exemption is not None and exemption.active) else ()
    gate_ok = result.ok or bool(waived)

    report = render_gate_report(result, exemption=exemption)

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
    failures = result.failures
    data: dict[str, Any] = {
        "base_collect": str(base_collect_path),
        "head_collect": str(head_collect_path),
        "base_leaf_count": base_total,
        "head_leaf_count": head_total,
        "failing_count": len(failures),
        "findings": [
            {
                "kind": f.kind,
                "leaf_name": f.leaf_name,
                "source_module": f.source_module,
                "detail": f.detail,
                "base_count": f.base_count,
                "head_count": f.head_count,
                "fails": f.fails_gate,
            }
            for f in result.findings
        ],
    }
    # The machine-readable line the review packet reads back out of this
    # job's log (issue #1686): emitted whenever an exemption was evaluated
    # (--pr given), whether or not it was granted, so the packet can tell
    # "waived N findings on this head" apart from "label present but the
    # run never applied it" and "label present but there was nothing to
    # waive". The same payload rides data["exemption"] for the printed-JSON
    # consumer.
    exemption_line = ""
    if exemption is not None:
        marker = exemption_log_marker(exemption, waived, head_sha=head_sha)
        data["exemption"] = json.loads(marker[len(EXEMPTION_LOG_MARKER) :])
        exemption_line = "\n" + marker
    else:
        data["exemption"] = None

    if gate_ok:
        reported = len(result.findings)
        suffix = (
            "multisets match"
            if reported == 0
            else f"{reported} reported finding(s), none enforced"
        )
        message = (
            f"collect-only-check: PASSED ({base_total} leaf names at base, "
            f"{head_total} at head; {suffix})"
        )
        if waived:
            assert exemption is not None  # non-empty waived implies active
            waived_lines = [
                f"  {f.kind}: {f.leaf_name}"
                + (f" (from {f.source_module})" if f.source_module else "")
                for f in waived
            ]
            message = (
                f"collect-only-check: PASSED ({base_total} leaf names at base, "
                f"{head_total} at head; {len(waived)} failing finding(s) WAIVED "
                f"by operator exemption label '{exemption.label}')\n"
                + "\n".join(waived_lines)
                + f"\nexemption: {exemption.detail}."
            )
        elif exemption is not None:
            message += f"\nexemption: {exemption.detail}." + (
                " Nothing to waive -- no enforced findings on this run "
                "(a stale label can be removed)."
                if exemption.active
                else ""
            )
        return CommandResult(True, message + exemption_line, data)

    finding_lines = [
        f"  {f.kind}: {f.leaf_name}" + (f" (from {f.source_module})" if f.source_module else "")
        for f in failures
    ]
    message = (
        f"collect-only-check: FAILED ({len(failures)} failing finding(s), "
        f"{len(result.findings) - len(failures)} reported)\n"
        + "\n".join(finding_lines)
        + f"\nBase: {base_total} leaf names, Head: {head_total} leaf names.\n"
    )
    if exemption is not None:
        message += f"exemption: {exemption.detail}.\n"
    message += (
        "Scoped verdict (issue #1538): removed leaves, missing sibling "
        "reappearances, and shrunken multiplicities fail; net-new leaves "
        "and grown multiplicities are reported but pass. A verbatim test "
        "relocation (same leaf name, different module path) should pass; a "
        "rename, deletion, or class-wrapping dodge should fail. See the "
        "gate report for details."
    )
    return CommandResult(False, message + exemption_line, data)
