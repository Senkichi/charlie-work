from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from . import CLI_NAME
from .config import ConfigError, find_config_path
from .doctor import run_doctor
from .fleet_dispatch import fleet_loop
from .fleet_paths import fleet_dir
from .fleet_registry import _load_registry, touch_repo
from .global_config import load_layered_config
from .github import GitHub, GitHubError
from .paths import RepoNotFoundError, find_repo_root, runtime_paths
from .workflow import CommandResult, OrchestratorApp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=CLI_NAME)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument(
        "--fleet-dir", type=str, default=None, help="Override fleet directory path"
    )
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
    doctor.add_argument(
        "--live",
        action="store_true",
        dest="live",
        help=(
            "Validate gh --json field lists against the live gh CLI by executing "
            "read-only queries. Off by default because it requires network access "
            "to GitHub."
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
    loop_merge_group = loop.add_mutually_exclusive_group()
    loop_merge_group.add_argument("--merge", action="store_true", dest="merge")
    loop_merge_group.add_argument("--no-merge", action="store_false", dest="merge")
    loop.set_defaults(merge=None)

    fleet = subparsers.add_parser("fleet")
    fleet_sub = fleet.add_subparsers(dest="fleet_command", required=True)
    fleet_sub.add_parser("status")

    fleet_work = fleet_sub.add_parser("work")
    fleet_work.add_argument("--limit", type=int, default=None)
    fleet_work.add_argument(
        "--repos",
        type=str,
        default=None,
        help=(
            "Comma-separated repo keys to process explicitly, e.g. "
            "'owner/repo1,owner/repo2'. Overrides the oldest-last_seen heuristic."
        ),
    )

    fleet_bash_rats = fleet_sub.add_parser("bash-rats")
    fleet_bash_rats.add_argument("--limit", type=int, default=None)
    fleet_bash_rats.add_argument(
        "--repos",
        type=str,
        default=None,
        help=(
            "Comma-separated repo keys to process explicitly, e.g. "
            "'owner/repo1,owner/repo2'. Overrides the oldest-last_seen heuristic."
        ),
    )
    fleet_bash_rats_merge_group = fleet_bash_rats.add_mutually_exclusive_group()
    fleet_bash_rats_merge_group.add_argument("--merge", action="store_true", dest="merge")
    fleet_bash_rats_merge_group.add_argument("--no-merge", action="store_false", dest="merge")
    fleet_bash_rats.set_defaults(merge=None)

    return parser


def build_app(args: argparse.Namespace) -> OrchestratorApp:
    repo_root = find_repo_root(args.repo, explicit=args.repo is not None)
    config = load_layered_config(repo_root, args.config, fleet_dir_override=args.fleet_dir)
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    gh = GitHub(repo_root=repo_root, dry_run=args.dry_run)
    touch_repo(args.fleet_dir, repo_root, paths, gh)
    return OrchestratorApp(
        repo_root, paths, config, gh, dry_run=args.dry_run, fleet_dir_override=args.fleet_dir
    )


def run_doctor_command(args: argparse.Namespace) -> CommandResult:
    repo_root = find_repo_root(args.repo, explicit=args.repo is not None)
    config_path = find_config_path(repo_root, args.config)
    config = load_layered_config(repo_root, args.config, fleet_dir_override=args.fleet_dir)
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    gh = GitHub(repo_root=repo_root, dry_run=args.dry_run)
    touch_repo(args.fleet_dir, repo_root, paths, gh)
    ok, checks = run_doctor(
        repo_root, paths, config, config_path, gh, adapter_probe=args.adapter_probe, live=args.live
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


def run_fleet_work(args: argparse.Namespace) -> CommandResult:
    """Run fleet work (dispatch-only) across all or selected registered repos.

    This is the fleet-wide version of the single-repo 'work' command:
    - Runs dispatch only (no review/merge stage)
    - Respects global concurrency budget via per-repo governor
    - Aggregates results into one consolidated attention digest
    """
    # Parse --repos into tuple if provided
    repos = tuple(args.repos.split(",")) if args.repos else None

    # Load global config for notifier integration (optional, may be None)
    try:
        global_config = load_layered_config(Path.cwd(), None, fleet_dir_override=args.fleet_dir)
    except (ConfigError, RepoNotFoundError):
        global_config = None

    return fleet_loop(
        fleet_dir_override=args.fleet_dir,
        global_config=global_config,
        repos=repos,
        limit=args.limit,
        merge=None,  # work-only doesn't use merge
        dry_run=args.dry_run,
        work_only=True,
    )


def run_fleet_bash_rats(args: argparse.Namespace) -> CommandResult:
    """Run fleet bash-rats (full loop) across all or selected registered repos.

    This is the fleet-wide version of the single-repo 'bash-rats' command:
    - Runs full loop (intake -> dispatch -> review -> merge)
    - Respects global concurrency budget via per-repo governor
    - Aggregates results into one consolidated attention digest
    """
    # Parse --repos into tuple if provided
    repos = tuple(args.repos.split(",")) if args.repos else None

    # Load global config for notifier integration (optional, may be None)
    try:
        global_config = load_layered_config(Path.cwd(), None, fleet_dir_override=args.fleet_dir)
    except (ConfigError, RepoNotFoundError):
        global_config = None

    return fleet_loop(
        fleet_dir_override=args.fleet_dir,
        global_config=global_config,
        repos=repos,
        limit=args.limit,
        merge=args.merge,
        dry_run=args.dry_run,
        work_only=False,
    )


def run_fleet_status(args: argparse.Namespace) -> CommandResult:
    """Run fleet status aggregation across all registered repos.

    This is a read-only command that:
    - Loads the fleet registry from fleet.json
    - For each registered repo, calls OrchestratorApp.status() with dry_run=True
    - Aggregates results keyed by repo_key (nameWithOwner)
    - Isolates per-repo errors (missing/broken repos) without aborting the whole aggregation

    Note: This command does not include per-worker health fields yet. That will be added
    in a follow-up issue (#167) once the worker health abstraction lands.
    """
    fleet_json_path = fleet_dir() / "fleet.json"
    registry = _load_registry(fleet_json_path)
    per_repo: dict[str, Any] = {}
    errors: list[dict[str, str]] = []

    for repo_key, entry in sorted(registry.get("repos", {}).items()):
        try:
            repo_root = Path(entry.get("repo_root"))
            if not repo_root.exists():
                raise RepoNotFoundError(f"Repo root does not exist: {repo_root}")

            config = load_layered_config(repo_root, None, fleet_dir_override=args.fleet_dir)
            paths = runtime_paths(repo_root, config.runtime.state_dir)
            gh = GitHub(repo_root=repo_root, dry_run=True)
            app = OrchestratorApp(repo_root, paths, config, gh, dry_run=True)
            result = app.status()
            per_repo[repo_key] = result.data
        except (RepoNotFoundError, ConfigError, GitHubError, OSError) as exc:
            errors.append({"repo_key": repo_key, "error": str(exc)})

    return CommandResult(
        ok=not errors,
        message=f"fleet status: {len(per_repo)} repo(s), {len(errors)} error(s)",
        data={"repos": per_repo, "errors": errors},
    )


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
        return app.loop(args.limit, merge=args.merge)
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
        elif args.command == "fleet":
            if args.fleet_command == "status":
                result = run_fleet_status(args)
            elif args.fleet_command == "work":
                result = run_fleet_work(args)
            elif args.fleet_command == "bash-rats":
                result = run_fleet_bash_rats(args)
            else:
                result = CommandResult(False, f"unknown fleet command: {args.fleet_command}", {})
        else:
            app = build_app(args)
            result = run_command(app, args)
    except RepoNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except GitHubError as exc:
        print(f"GitHub error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"OS error: {exc}", file=sys.stderr)
        return 2
    except (ConfigError, ValueError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except yaml.YAMLError as exc:
        print(f"YAML error: {exc}", file=sys.stderr)
        return 2

    # Custom human-readable rendering for fleet commands
    if args.command == "fleet" and not json_output:
        print(result.message)
        if args.fleet_command == "status":
            repos = result.data.get("repos", {})
            for repo_key, repo_data in sorted(repos.items()):
                ready = repo_data.get("ready_issue_count", 0)
                active = repo_data.get("active_issue_count", 0)
                blocked = len(repo_data.get("blocked", []))
                stalled = len(repo_data.get("stalled", []))
                print(
                    f"  {repo_key}: {ready} ready, {active} active, {blocked} blocked, {stalled} stalled"
                )
            errors = result.data.get("errors", [])
            if errors:
                print("Errors:")
                for error in errors:
                    print(f"  {error['repo_key']}: {error['error']}")
        elif args.fleet_command in ("work", "bash-rats"):
            repos = result.data.get("repos", {})
            for repo_key, repo_data in sorted(repos.items()):
                # repo_data now includes the ok field from fleet_dispatch
                ok = repo_data.get("ok", True)
                status = "OK" if ok else "FAILED"
                print(f"  {repo_key}: {status}")
            digest = result.data.get("digest", {})
            event_count = digest.get("count", 0)
            orphan_sweep_calls = digest.get("orphan_sweep_calls", 0)
            print(
                f"  Digest: {event_count} attention event(s), {orphan_sweep_calls} orphan sweep call(s)"
            )
    else:
        print_result(result, json_output=args.json_output)

    return 0 if result.ok else 1
