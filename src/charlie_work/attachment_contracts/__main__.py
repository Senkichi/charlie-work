"""CLI: scan | baseline | check-file | check-tree | backtest.

Exit codes:
- `scan`: 0 clean; 1 if any file failed to parse (G6 — fail toward CI).
- `baseline`: 0 on success; 1 if `--ratchet` was requested but no baseline
  file exists yet to ratchet.
- `check-file` / `check-tree`: 0 clean; 1 if any Finding is `block` or
  `error`. `check-tree --report-only` always exits 0 (Week-1 shadow mode).
  # WEEK-2: flip enforcement by deleting `--report-only` from the CI workflow
  # invocation (the exit-code logic below already supports it: `return 1 if
  # blocking else 0` unconditionally).
- `backtest`: 0 if the backtest passed its gate criteria, else 1.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

from charlie_work.attachment_contracts.archetypes import scan_tree
from charlie_work.attachment_contracts.backtest import ANCHOR_SHAS, run_backtest, write_report
from charlie_work.attachment_contracts.baseline import (
    BASELINE_FILENAME,
    TamperError,
    compare,
    dump,
    load,
)
from charlie_work.attachment_contracts.baseline import (
    generate as generate_baseline,
)
from charlie_work.attachment_contracts.baseline import loads as load_baseline_text
from charlie_work.attachment_contracts.check import check_file, check_tree
from charlie_work.attachment_contracts.excludes import load_excludes
from charlie_work.attachment_contracts.model import Finding
from charlie_work.attachment_contracts.outliers import FLOOR, saturate_all
from charlie_work.subprocess_runner import no_console_window_kwargs

PACKAGE_VERSION = "0.1.0"


def _finding_to_dict(f: Finding) -> dict[str, object]:
    return {
        "severity": f.severity,
        "file": f.file,
        "identity": f.identity,
        "message": f.message,
        "redirect": f.redirect,
    }


def _print_findings(findings: list[Finding]) -> None:
    print(json.dumps([_finding_to_dict(f) for f in findings], indent=1, sort_keys=True))


def _is_blocking(findings: list[Finding]) -> bool:
    return any(f.severity in ("block", "error") for f in findings)


def _github_annotation(f: Finding) -> str:
    level = "error" if f.severity in ("block", "error") else "warning"
    return f"::{level} file={f.file}::{f.identity}: {f.message}"


def _cmd_scan(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    excludes = load_excludes(root)
    scan = scan_tree(root, excludes)
    document = {
        "root": scan.root,
        "point_count": len(scan.points),
        "parse_failures": list(scan.parse_failures),
        "points": [
            {
                "kind": p.kind,
                "identity": p.identity,
                "file": p.file,
                "member_count": p.member_count,
                "is_linear_ledger": p.is_linear_ledger,
                "is_structurally_trivial": p.is_structurally_trivial,
            }
            for p in scan.points
        ],
    }
    print(json.dumps(document, indent=1, sort_keys=True))
    return 1 if scan.parse_failures else 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    excludes = load_excludes(root)
    scan = scan_tree(root, excludes)
    kinds = sorted({p.kind for p in scan.points})
    verdicts = saturate_all(scan.points, kinds)
    baseline_path = root / BASELINE_FILENAME

    if args.ratchet:
        if not baseline_path.is_file():
            print(f"error: no baseline at {baseline_path} to ratchet", file=sys.stderr)
            return 1
        document = load(baseline_path)
        _findings, ratcheted = compare(verdicts, document)
        dump(ratcheted, baseline_path)
        print(f"ratcheted baseline written: {baseline_path} ({len(ratcheted['entries'])} entries)")
        return 0

    generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    document = generate_baseline(
        verdicts,
        generated_by=f"charlie_work.attachment_contracts {PACKAGE_VERSION}",
        generated_at=generated_at,
        floor=FLOOR,
    )
    dump(document, baseline_path)
    print(f"baseline written: {baseline_path} ({len(document['entries'])} entries)")
    return 0


def _cmd_check_file(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    findings = check_file(args.path, root)
    _print_findings(findings)
    return 1 if _is_blocking(findings) else 0


def _load_previous_baseline_document(root: Path, base_ref: str) -> dict[str, object] | None:
    """Fetch `.attachment-budgets.json` as it read at `base_ref`, for the G4
    diff-based ratchet-tamper guard (finding #1). Returns None -- meaning
    "nothing to diff against, skip that check for this run" -- whenever it
    genuinely can't be resolved: the ref doesn't exist, the file didn't exist
    yet at that ref (freeze-on-adopt's first commit), or this isn't a git
    checkout at all. Never raises: the base-ref lookup is a bonus check, not
    a precondition for `check-tree` to run at all.
    """
    try:
        result = subprocess.run(
            ["git", "show", f"{base_ref}:{BASELINE_FILENAME}"],
            cwd=str(root),
            capture_output=True,
            text=True,
            **no_console_window_kwargs(),
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    try:
        return load_baseline_text(result.stdout)
    except TamperError:
        return None


def _cmd_check_tree(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    previous_document = (
        _load_previous_baseline_document(root, args.base_ref) if args.base_ref else None
    )
    findings = check_tree(root, previous_baseline_document=previous_document)
    if args.github_annotations:
        for f in findings:
            print(_github_annotation(f))
    _print_findings(findings)
    if args.report_only:
        return 0
    return 1 if _is_blocking(findings) else 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    repo_path = Path(args.repo).resolve()
    verdict = run_backtest(
        repo_path, months=args.months, anchor_shas=ANCHOR_SHAS, branch=args.branch
    )
    out_dir = Path(args.out_dir).resolve()
    md_path, json_path = write_report(verdict, out_dir)
    print(f"backtest {'PASS' if verdict.passed else 'FAIL'}: {md_path}, {json_path}")
    return 0 if verdict.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m charlie_work.attachment_contracts")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Scan the tree and print detected attachment points.")
    p_scan.add_argument("--root", default=".")
    p_scan.set_defaults(func=_cmd_scan)

    p_baseline = sub.add_parser("baseline", help="Generate or ratchet .attachment-budgets.json.")
    p_baseline.add_argument("--root", default=".")
    p_baseline.add_argument("--ratchet", action="store_true")
    p_baseline.set_defaults(func=_cmd_baseline)

    p_check_file = sub.add_parser("check-file", help="Check a single file against the baseline.")
    p_check_file.add_argument("path")
    p_check_file.add_argument("--root", default=".")
    p_check_file.set_defaults(func=_cmd_check_file)

    p_check_tree = sub.add_parser("check-tree", help="Full-tree check against the baseline.")
    p_check_tree.add_argument("--root", default=".")
    p_check_tree.add_argument("--report-only", action="store_true")
    p_check_tree.add_argument("--github-annotations", action="store_true")
    p_check_tree.add_argument(
        "--base-ref",
        default=None,
        help=(
            "git ref to diff the committed baseline against for the ratchet-tamper "
            "guard (finding #1). Omit to skip that check (e.g. outside CI)."
        ),
    )
    p_check_tree.set_defaults(func=_cmd_check_tree)

    p_backtest = sub.add_parser("backtest", help="G1 positive-control backtest over git history.")
    p_backtest.add_argument("--repo", default=".")
    p_backtest.add_argument("--months", type=int, default=6)
    p_backtest.add_argument("--branch", default="main")
    p_backtest.add_argument("--out-dir", default="docs/plans")
    p_backtest.set_defaults(func=_cmd_backtest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
