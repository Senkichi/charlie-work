from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import CLI_NAME
from .config import find_config_path, load_config
from .doctor import run_doctor
from .github import GitHub, GitHubError
from .paths import find_repo_root, runtime_paths
from .workflow import CommandResult, OrchestratorApp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=CLI_NAME)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("roll-call")
    doctor = subparsers.add_parser("doctor")
    doctor.add_argument(
        "--adapter-probe",
        action="store_true",
        dest="adapter_probe",
        help=(
            "Also execute the configured worker adapter's CLI probe (e.g. "
            "'devin --version') and surface stale/failed launched sessions. "
            "Off by default because probes run external binaries."
        ),
    )
    subparsers.add_parser("bootstrap-labels")
    subparsers.add_parser("intake")

    reconcile = subparsers.add_parser("mop-up")
    reconcile.add_argument(
        "--fix",
        action="store_true",
        help=(
            "Apply label/state repairs for detected drift. Without this flag "
            "reconcile is strictly read-only."
        ),
    )

    dispatch = subparsers.add_parser("work")
    dispatch.add_argument("--limit", type=int, default=None)
    dispatch.add_argument(
        "--issues",
        type=str,
        default=None,
        help=(
            "Comma-separated issue numbers to dispatch explicitly, e.g. "
            "'565,570,572'. Overrides the newest-first heuristic so the operator "
            "can dispatch dependency-ordered waves (foundations before leaves). "
            "Numbers that are not currently dispatchable are skipped."
        ),
    )

    review = subparsers.add_parser("why-charlie-hate")
    review.add_argument("--pr", type=int, required=True)
    cross_family_group = review.add_mutually_exclusive_group()
    cross_family_group.add_argument(
        "--cross-family", action="store_const", const=True, dest="cross_family", default=None
    )
    cross_family_group.add_argument(
        "--no-cross-family", action="store_const", const=False, dest="cross_family"
    )

    spec_review = subparsers.add_parser("why-charlie-hate-spec")
    spec_review.add_argument("--file", type=Path, required=True, dest="spec_file")

    record = subparsers.add_parser("verdict")
    record.add_argument("--pr", type=int, required=True)
    record.add_argument(
        "--decision", choices=["approved", "request_changes", "blocked"], required=True
    )
    record.add_argument("--summary", default="")
    record.add_argument("--summary-file", type=Path, default=None)
    record.add_argument("--comment", action="store_true")

    merge_ready = subparsers.add_parser("ship-it")
    merge_ready.add_argument("--pr", type=int, required=True)
    merge_group = merge_ready.add_mutually_exclusive_group()
    merge_group.add_argument("--merge", action="store_true", dest="merge")
    merge_group.add_argument("--no-merge", action="store_false", dest="merge")
    merge_ready.set_defaults(merge=None)

    loop = subparsers.add_parser("bash-rats")
    loop.add_argument("--limit", type=int, default=None)

    return parser


def build_app(args: argparse.Namespace) -> OrchestratorApp:
    repo_root = find_repo_root(args.repo)
    config = load_config(find_config_path(repo_root, args.config))
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    gh = GitHub(repo_root=repo_root, dry_run=args.dry_run)
    return OrchestratorApp(repo_root, paths, config, gh)


def run_doctor_command(args: argparse.Namespace) -> CommandResult:
    repo_root = find_repo_root(args.repo)
    config_path = find_config_path(repo_root, args.config)
    config = load_config(config_path)
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    gh = GitHub(repo_root=repo_root, dry_run=args.dry_run)
    ok, checks = run_doctor(
        repo_root, paths, config, config_path, gh, adapter_probe=args.adapter_probe
    )
    failed = [check for check in checks if not check.ok]
    message = (
        "doctor: all checks passed"
        if ok and not failed
        else f"doctor: {len(failed)} finding(s)"
        if ok
        else f"doctor: {len(failed)} finding(s), at least one blocking"
    )
    return CommandResult(ok, message, {"checks": [check.to_dict() for check in checks]})


def run_command(app: OrchestratorApp, args: argparse.Namespace) -> CommandResult:
    if args.command == "roll-call":
        return app.status()
    if args.command == "bootstrap-labels":
        return app.bootstrap_labels()
    if args.command == "intake":
        return app.intake()
    if args.command == "mop-up":
        return app.reconcile(fix=args.fix)
    if args.command == "work":
        return app.dispatch(args.limit, only_issues=args.issues)
    if args.command == "why-charlie-hate":
        return app.review(args.pr, cross_family=args.cross_family)
    if args.command == "why-charlie-hate-spec":
        return app.spec_review(args.spec_file)
    if args.command == "verdict":
        return app.record_review(
            args.pr,
            args.decision,
            summary=args.summary,
            summary_file=args.summary_file,
            comment=args.comment,
        )
    if args.command == "ship-it":
        return app.merge_ready(args.pr, merge=args.merge)
    if args.command == "bash-rats":
        return app.loop(args.limit)
    return CommandResult(False, f"unknown command: {args.command}", {})


def print_result(result: CommandResult, *, json_output: bool) -> None:
    payload: dict[str, Any] = {"ok": result.ok, "message": result.message, "data": result.data}
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return
    print(result.message)
    if result.data:
        print(json.dumps(result.data, indent=2, sort_keys=True, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    json_output = "--json" in raw_argv
    args = parser.parse_args([arg for arg in raw_argv if arg != "--json"])
    args.json_output = json_output or args.json_output
    try:
        if args.command == "doctor":
            result = run_doctor_command(args)
        else:
            app = build_app(args)
            result = run_command(app, args)
    except GitHubError as exc:
        print(f"GitHub error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"OS error: {exc}", file=sys.stderr)
        return 2
    print_result(result, json_output=args.json_output)
    return 0 if result.ok else 1
