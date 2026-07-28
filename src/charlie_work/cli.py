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
from .fleet_dispatch import compute_api_worker_fleet_report, fleet_loop, run_fleet_supervise
from .fleet_paths import fleet_dir
from .fleet_registry import _load_registry, touch_repo, count_fleet_runners
from .global_config import load_layered_config
from .github import GitHub, GitHubError
from .logging_setup import configure_logging
from .notify import AttentionDigest, AttentionEntry, emit_digest
from .paths import RepoNotFoundError, find_repo_root, runtime_paths
from .state import StateLockBusy, load_state_locked, utc_now
from .supervise import orchestrator_root, self_deploy
from .runner_allocation import plan_summary
from .runner_allocation_pass import run_allocation_pass
from .runner_slots import CLI_ALLOCATION_SOURCE
from .runners import (
    decide_autoscale,
    ensure_runners_started,
    format_runner_pool_state,
    FleetTotals,
    is_in_cooldown,
    is_pool_idle_for_minutes,
    observe_runner_pool,
    scale_down_idle_runners,
    ScaleAction,
)
from .worktree import clean_worktrees
from .workflow import CommandResult, OrchestratorApp


def _add_dry_run(parser: argparse.ArgumentParser) -> None:
    """Add a subcommand-level ``--dry-run`` that does not clobber the global one.

    ``--dry-run`` also exists on the top-level parser, so an operator may write it
    either before or after the subcommand. Both must work, and the plain idiom does
    not: without ``SUPPRESS`` the subparser applies its own ``False`` default *after*
    the top-level flag was parsed, silently overwriting it. ``charlie --dry-run
    runners allocate`` therefore ran for real — and for ``allocate`` "for real"
    means starting and terminating live CI listeners during what the operator
    believes is a simulation. Observed on the live host: a global-flag dry run
    launched a listener and reported its PID.

    ``SUPPRESS`` makes the subparser set the attribute only when the flag is
    actually present, so the global value stands when it is not. The top-level flag
    keeps its ``False`` default, so ``args.dry_run`` always exists. Route every new
    subcommand-level flag through here rather than repeating the idiom.
    """
    parser.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=CLI_NAME)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--verbose", action="store_true", help="Enable debug-level logging")
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

    claim = subparsers.add_parser("claim")
    claim.add_argument("issue", type=int, help="Issue number to claim or release")
    claim.add_argument(
        "--release",
        action="store_true",
        help="Release an existing operator claim instead of recording one",
    )

    reconcile = subparsers.add_parser("mop-up")
    reconcile.add_argument(
        "--fix",
        action="store_true",
        help=(
            "Apply label/state repairs for detected drift. Without this flag "
            "reconcile is strictly read-only."
        ),
    )

    subparsers.add_parser("review-queue")

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
    record.add_argument("--reviewed-head", default=None)
    record.add_argument("--comment", action="store_true")

    unescalate = subparsers.add_parser("unescalate")
    unescalate.add_argument("--pr", type=int, default=None)
    unescalate.add_argument("--issue", type=int, default=None)
    _add_dry_run(unescalate)

    merge_ready = subparsers.add_parser("ship-it")
    merge_ready.add_argument("--pr", type=int, required=True)
    merge_group = merge_ready.add_mutually_exclusive_group()
    merge_group.add_argument("--merge", action="store_true", dest="merge")
    merge_group.add_argument("--no-merge", action="store_false", dest="merge")
    merge_ready.set_defaults(merge=None)

    loop = subparsers.add_parser("bash-rats")
    loop.add_argument("--limit", type=int, default=None)
    loop.add_argument(
        "--once",
        action="store_true",
        default=False,
        help="Run a single pass and exit (preserves legacy behavior).",
    )
    loop.add_argument(
        "--poll-interval",
        type=int,
        default=None,
        dest="poll_interval",
        help="Override supervisor poll_interval_seconds from config.",
    )
    loop.add_argument(
        "--max-runtime",
        type=int,
        default=None,
        dest="max_runtime",
        help="Override supervisor max_runtime_minutes from config (0 = unlimited).",
    )
    loop_merge_group = loop.add_mutually_exclusive_group()
    loop_merge_group.add_argument("--merge", action="store_true", dest="merge")
    loop_merge_group.add_argument("--no-merge", action="store_false", dest="merge")
    loop.set_defaults(merge=None)

    fleet = subparsers.add_parser("fleet")
    fleet_sub = fleet.add_subparsers(dest="fleet_command", required=True)
    fleet_sub.add_parser("status")
    fleet_sub.add_parser("review-queue")

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

    fleet_supervise = fleet_sub.add_parser("supervise")
    fleet_supervise.add_argument("--limit", type=int, default=None)
    fleet_supervise.add_argument(
        "--repos",
        type=str,
        default=None,
        help=(
            "Comma-separated repo keys to process explicitly, e.g. "
            "'owner/repo1,owner/repo2'. Overrides the oldest-last_seen heuristic."
        ),
    )
    fleet_supervise.add_argument(
        "--poll-interval",
        type=int,
        default=None,
        dest="poll_interval",
        help="Override supervisor poll_interval_seconds from config.",
    )
    fleet_supervise.add_argument(
        "--max-runtime",
        type=int,
        default=None,
        dest="max_runtime",
        help="Override supervisor max_runtime_minutes from config (0 = unlimited).",
    )
    fleet_supervise_merge_group = fleet_supervise.add_mutually_exclusive_group()
    fleet_supervise_merge_group.add_argument("--merge", action="store_true", dest="merge")
    fleet_supervise_merge_group.add_argument("--no-merge", action="store_false", dest="merge")
    fleet_supervise.set_defaults(merge=None)

    runners = subparsers.add_parser("runners")
    runners_sub = runners.add_subparsers(dest="runners_command", required=True)
    runners_sub.add_parser("status")
    ensure_started_parser = runners_sub.add_parser("ensure-started")
    _add_dry_run(ensure_started_parser)
    scale_down_parser = runners_sub.add_parser("scale-down")
    _add_dry_run(scale_down_parser)
    autoscale_parser = runners_sub.add_parser("autoscale")
    _add_dry_run(autoscale_parser)
    autoscale_parser.add_argument(
        "--fleet-wide", action="store_true", help="Use fleet-wide runner counts for guardrails"
    )
    allocate_parser = runners_sub.add_parser(
        "allocate",
        help=(
            "Rebalance this host's running runner listeners across every repo "
            "with runners registered under managed_root, by live queue demand"
        ),
    )
    _add_dry_run(allocate_parser)

    subparsers.add_parser("worktree-clean")

    return parser


def build_app(args: argparse.Namespace) -> OrchestratorApp:
    repo_root = find_repo_root(args.repo, explicit=args.repo is not None)
    config = load_layered_config(repo_root, args.config, fleet_dir_override=args.fleet_dir)
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    gh = GitHub(repo_root=repo_root, runtime=config.runtime, dry_run=args.dry_run)
    touch_repo(args.fleet_dir, repo_root, paths, gh)
    return OrchestratorApp(
        repo_root, paths, config, gh, dry_run=args.dry_run, fleet_dir_override=args.fleet_dir
    )


def run_doctor_command(args: argparse.Namespace) -> CommandResult:
    repo_root = find_repo_root(args.repo, explicit=args.repo is not None)
    config_path = find_config_path(repo_root, args.config)
    config = load_layered_config(repo_root, args.config, fleet_dir_override=args.fleet_dir)
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    gh = GitHub(repo_root=repo_root, runtime=config.runtime, dry_run=args.dry_run)
    touch_repo(args.fleet_dir, repo_root, paths, gh)
    ok, checks = run_doctor(
        repo_root,
        paths,
        config,
        config_path,
        gh,
        adapter_probe=args.adapter_probe,
        live=args.live,
        fleet_dir_override=args.fleet_dir,
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


def run_worktree_clean_command(args: argparse.Namespace) -> CommandResult:
    repo_root = find_repo_root(args.repo, explicit=args.repo is not None)
    config = load_layered_config(repo_root, args.config, fleet_dir_override=args.fleet_dir)
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    gh = GitHub(repo_root=repo_root, runtime=config.runtime, dry_run=args.dry_run)
    state = load_state_locked(paths.state_file)
    result = clean_worktrees(
        repo_root,
        paths.root / "worktrees",
        state,
        config,
        gh,
        dry_run=args.dry_run,
    )
    return CommandResult(result.ok, result.message, result.data)


def run_fleet_work(args: argparse.Namespace) -> CommandResult:
    """Run fleet work (dispatch-only) across all or selected registered repos.

    This is the fleet-wide version of the single-repo 'work' command:
    - Runs dispatch only (no review/merge stage)
    - Respects global concurrency budget via per-repo governor
    - Aggregates results into one consolidated attention digest
    """
    # Parse --repos into tuple if provided
    repos = tuple(args.repos.split(",")) if args.repos else None

    # Load global config for notifier integration (optional, may be None).
    # A failure here turns off every config-gated fleet behavior (notify and the
    # runner prologues), so it is reported rather than swallowed.
    try:
        global_config = load_layered_config(
            Path.cwd(),
            None,
            fleet_dir_override=args.fleet_dir,
            require_global=True,
        )
    except (ConfigError, RepoNotFoundError) as exc:
        print(f"config load failed, fleet running without global config: {exc}", flush=True)
        # The global layer is required, but the per-repo config is still valid
        # and must not be discarded with it -- discarding both regresses the
        # #623 silent-disable failure (every per-repo knob reverting to its
        # dataclass default while passes keep reporting success). Reload
        # without the global requirement so per-repo settings survive; only
        # fall back to None if the per-repo load itself fails.
        try:
            global_config = load_layered_config(
                Path.cwd(), None, fleet_dir_override=args.fleet_dir
            )
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

    # Load global config for notifier integration (optional, may be None).
    # A failure here turns off every config-gated fleet behavior (notify and the
    # runner prologues), so it is reported rather than swallowed.
    try:
        global_config = load_layered_config(
            Path.cwd(),
            None,
            fleet_dir_override=args.fleet_dir,
            require_global=True,
        )
    except (ConfigError, RepoNotFoundError) as exc:
        print(f"config load failed, fleet running without global config: {exc}", flush=True)
        # The global layer is required, but the per-repo config is still valid
        # and must not be discarded with it -- discarding both regresses the
        # #623 silent-disable failure (every per-repo knob reverting to its
        # dataclass default while passes keep reporting success). Reload
        # without the global requirement so per-repo settings survive; only
        # fall back to None if the per-repo load itself fails.
        try:
            global_config = load_layered_config(
                Path.cwd(), None, fleet_dir_override=args.fleet_dir
            )
        except (ConfigError, RepoNotFoundError):
            global_config = None

    # Self-deploy before running the pass: FF-pull origin/main and sync
    # dependencies when pyproject.toml/uv.lock changed. Non-fatal on a
    # diverged or dirty tree.
    deploy = self_deploy(
        orchestrator_root(), fleet_dir_override=args.fleet_dir, dry_run=args.dry_run
    )
    if not deploy.ok:
        print(f"self-deploy skipped: {deploy.error}", flush=True)
        notify_config = getattr(global_config, "notify", None) if global_config else None
        if (
            deploy.alertable
            and notify_config is not None
            and getattr(notify_config, "enabled", False)
        ):
            attention_digest = AttentionDigest(
                generated_at=utc_now(),
                repo="fleet",
                transitions=(
                    AttentionEntry(
                        issue_number=-1,
                        adapter_kind="self-deploy",
                        health="ERROR",
                        previous_health=None,
                        last_log_line=deploy.error,
                        pid=None,
                    ),
                ),
            )
            emit_digest(notify_config, attention_digest)
    elif deploy.previewed:
        print(f"self-deploy: {deploy.message}", flush=True)
    elif deploy.synced:
        print(f"self-deploy: {deploy.message}", flush=True)
    elif deploy.venv_repaired:
        print(f"self-deploy: {deploy.message}", flush=True)
        notify_config = getattr(global_config, "notify", None) if global_config else None
        if notify_config is not None and getattr(notify_config, "enabled", False):
            attention_digest = AttentionDigest(
                generated_at=utc_now(),
                repo="fleet",
                transitions=(
                    AttentionEntry(
                        issue_number=-1,
                        adapter_kind="self-deploy",
                        health="REPAIRED",
                        previous_health=None,
                        last_log_line=deploy.message,
                        pid=None,
                    ),
                ),
            )
            emit_digest(notify_config, attention_digest)

    return fleet_loop(
        fleet_dir_override=args.fleet_dir,
        global_config=global_config,
        repos=repos,
        limit=args.limit,
        merge=args.merge,
        dry_run=args.dry_run,
        work_only=False,
    )


def run_fleet_supervise_command(args: argparse.Namespace) -> CommandResult:
    """Run the continuous fleet supervisor.

    This is the fleet-wide equivalent of the single-repo 'bash-rats' supervisor:
    repeated fleet passes separated by configurable sleep, honoring the
    supervisor section of the config (poll_interval_seconds, active_cooldown_seconds,
    max_runtime_minutes).
    """
    # Parse --repos into tuple if provided
    repos = tuple(args.repos.split(",")) if args.repos else None

    return run_fleet_supervise(
        fleet_dir_override=args.fleet_dir,
        repos=repos,
        limit=args.limit,
        merge=args.merge,
        dry_run=args.dry_run,
        poll_interval_override=args.poll_interval,
        max_runtime_override=args.max_runtime,
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
            gh = GitHub(repo_root=repo_root, runtime=config.runtime, dry_run=True)
            app = OrchestratorApp(repo_root, paths, config, gh, dry_run=True)
            result = app.status()
            per_repo[repo_key] = result.data
        except (RepoNotFoundError, ConfigError, GitHubError, OSError) as exc:
            errors.append({"repo_key": repo_key, "error": str(exc)})

    # api-worker fleet report line (issue #483): read-only, never raises.
    api_worker_report = compute_api_worker_fleet_report(fleet_dir_override=args.fleet_dir)

    return CommandResult(
        ok=not errors,
        message=f"fleet status: {len(per_repo)} repo(s), {len(errors)} error(s)",
        data={
            "repos": per_repo,
            "errors": errors,
            "api_worker_report": api_worker_report.to_dict()
            if api_worker_report is not None
            else None,
        },
    )


def run_fleet_review_queue(args: argparse.Namespace) -> CommandResult:
    """Run fleet review-queue aggregation across all registered repos.

    This is a read-only command that:
    - Loads the fleet registry from fleet.json
    - For each registered repo, calls OrchestratorApp.review_queue() with dry_run=True
    - Aggregates per-repo queue entries keyed by repo_key (nameWithOwner)
    - Isolates per-repo errors (missing/broken repos) without aborting aggregation
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
            gh = GitHub(repo_root=repo_root, runtime=config.runtime, dry_run=True)
            app = OrchestratorApp(repo_root, paths, config, gh, dry_run=True)
            result = app.review_queue()
            per_repo[repo_key] = result.data
        except (RepoNotFoundError, ConfigError, GitHubError, OSError) as exc:
            errors.append({"repo_key": repo_key, "error": str(exc)})

    return CommandResult(
        ok=not errors,
        message=f"fleet review queue: {len(per_repo)} repo(s), {len(errors)} error(s)",
        data={"repos": per_repo, "errors": errors},
    )


def run_runners_status(args: argparse.Namespace) -> CommandResult:
    """Run runner pool status for the current repository.

    This is a read-only command that:
    - Loads the runner scaling configuration
    - Observes the runner pool state via GitHub API and host metrics
    - Returns formatted pool state with pressure classification
    - Saves pool samples for idle detection

    Returns an error if the runner_scaling feature is not enabled.
    """
    repo_root = find_repo_root(args.repo, explicit=args.repo is not None)
    config = load_layered_config(repo_root, args.config, fleet_dir_override=args.fleet_dir)
    paths = runtime_paths(repo_root, config.runtime.state_dir)

    if not config.runner_scaling.enabled:
        return CommandResult(
            ok=False,
            message="runner_scaling feature is not enabled in config",
            data={},
        )

    gh = GitHub(repo_root=repo_root, runtime=config.runtime, dry_run=args.dry_run)

    try:
        pool_state = observe_runner_pool(
            gh, config.runner_scaling, state_dir=paths.root, dry_run=args.dry_run
        )
        formatted = format_runner_pool_state(pool_state)
        return CommandResult(
            ok=True,
            message="runners status complete",
            data=formatted,
        )
    except GitHubError as exc:
        return CommandResult(
            ok=False,
            message=f"GitHub API error: {exc}",
            data={},
        )
    except Exception as exc:
        return CommandResult(
            ok=False,
            message=f"runners status failed: {exc}",
            data={},
        )


def run_runners_ensure_started(args: argparse.Namespace) -> CommandResult:
    """Ensure all configured managed runners are running.

    This command:
    - Loads the runner scaling configuration
    - Discovers managed runners under managed_root
    - Relaunches any configured-but-not-running managed runner
    - Returns the number of runners started and status messages

    Returns an error if the runner_scaling feature is not enabled.
    """
    repo_root = find_repo_root(args.repo, explicit=args.repo is not None)
    config = load_layered_config(repo_root, args.config, fleet_dir_override=args.fleet_dir)

    if not config.runner_scaling.enabled:
        return CommandResult(
            ok=False,
            message="runner_scaling feature is not enabled in config",
            data={},
        )

    if not config.runner_scaling.managed_root:
        return CommandResult(
            ok=False,
            message="runner_scaling.managed_root is not configured",
            data={},
        )

    managed_root = Path(config.runner_scaling.managed_root)
    if not managed_root.exists():
        return CommandResult(
            ok=False,
            message=f"managed_root does not exist: {managed_root}",
            data={},
        )

    # Use subparser-specific dry_run flag if available, otherwise fall back to global
    dry_run = getattr(args, "dry_run", False)

    started_count, messages = ensure_runners_started(
        managed_root,
        config.runner_scaling.runner_dir_prefix,
        config.runner_scaling,
        dry_run=dry_run,
    )

    return CommandResult(
        ok=True,
        message=f"runners ensure-started: {started_count} runner(s) started",
        data={"started_count": started_count, "messages": messages},
    )


def run_runners_scale_down(args: argparse.Namespace) -> CommandResult:
    """Scale down idle runners by gracefully removing them.

    This command:
    - Loads the runner scaling configuration
    - Checks if the pool has been idle for the required duration
    - Checks if we are in the cooldown period
    - Gracefully removes one idle runner if conditions are met
    - Returns the number of runners removed and any error messages

    Returns an error if the runner_scaling feature is not enabled.
    """
    repo_root = find_repo_root(args.repo, explicit=args.repo is not None)
    config = load_layered_config(repo_root, args.config, fleet_dir_override=args.fleet_dir)
    paths = runtime_paths(repo_root, config.runtime.state_dir)

    if not config.runner_scaling.enabled:
        return CommandResult(
            ok=False,
            message="runner_scaling feature is not enabled in config",
            data={},
        )

    if not config.runner_scaling.managed_root:
        return CommandResult(
            ok=False,
            message="runner_scaling.managed_root is not configured",
            data={},
        )

    managed_root = Path(config.runner_scaling.managed_root)
    if not managed_root.exists():
        return CommandResult(
            ok=False,
            message=f"managed_root does not exist: {managed_root}",
            data={},
        )

    # Use subparser-specific dry_run flag if available, otherwise fall back to global
    dry_run = getattr(args, "dry_run", False)

    gh = GitHub(repo_root=repo_root, runtime=config.runtime, dry_run=dry_run)

    removed_count, errors = scale_down_idle_runners(
        managed_root,
        config.runner_scaling.runner_dir_prefix,
        gh,
        config.runner_scaling,
        paths.root,
        dry_run=dry_run,
    )

    return CommandResult(
        ok=removed_count > 0 or not errors,
        message=f"runners scale-down: {removed_count} runner(s) removed",
        data={"removed_count": removed_count, "errors": errors},
    )


def run_runners_autoscale(args: argparse.Namespace) -> CommandResult:
    """Run autoscale decision and execute scale actions.

    This command:
    - Loads the runner scaling configuration
    - Observes the current runner pool state
    - Optionally loads fleet-wide runner totals for cross-repo guardrails
    - Runs the pure decision function to determine scale action
    - Executes scale actions (provision or scale-down) if not in dry-run mode
    - Returns the decision and execution result

    Returns an error if the runner_scaling feature is not enabled.
    """
    repo_root = find_repo_root(args.repo, explicit=args.repo is not None)
    config = load_layered_config(repo_root, args.config, fleet_dir_override=args.fleet_dir)
    paths = runtime_paths(repo_root, config.runtime.state_dir)

    if not config.runner_scaling.enabled:
        return CommandResult(
            ok=False,
            message="runner_scaling feature is not enabled in config",
            data={},
        )

    # Use subparser-specific dry_run flag if available, otherwise fall back to global
    dry_run = getattr(args, "dry_run", False)
    fleet_wide = getattr(args, "fleet_wide", False)

    gh = GitHub(repo_root=repo_root, runtime=config.runtime, dry_run=dry_run)

    # Observe current pool state
    state = observe_runner_pool(gh, config.runner_scaling, state_dir=paths.root, dry_run=dry_run)

    # Load fleet-wide totals if requested
    fleet_totals: FleetTotals | None = None
    skipped_repos: list[str] = []
    if fleet_wide:
        total_runners, total_busy_runners, skipped_repos = count_fleet_runners(
            args.fleet_dir, runtime=config.runtime
        )
        fleet_totals = FleetTotals(
            total_runners=total_runners,
            total_busy_runners=total_busy_runners,
        )

    # Check cooldown and idle duration
    in_cooldown = is_in_cooldown(paths.root, config.runner_scaling.cooldown_minutes)
    is_idle_for_duration = is_pool_idle_for_minutes(
        paths.root, config.runner_scaling.idle_scale_down_minutes
    )

    # Run the pure decision function
    decision = decide_autoscale(
        state,
        config.runner_scaling,
        fleet_totals=fleet_totals,
        in_cooldown=in_cooldown,
        is_idle_for_duration=is_idle_for_duration,
    )

    # In dry-run mode, just return the decision
    if dry_run:
        return CommandResult(
            ok=True,
            message=f"autoscale decision: {decision.action.value}({decision.count}) - {decision.reason}",
            data={
                "decision": {
                    "action": decision.action.value,
                    "count": decision.count,
                    "reason": decision.reason,
                },
                "state": format_runner_pool_state(state),
                "fleet_totals": {
                    "total_runners": fleet_totals.total_runners if fleet_totals else 0,
                    "total_busy_runners": fleet_totals.total_busy_runners if fleet_totals else 0,
                    # Repos whose runner count could not be read are an
                    # undercount in the guardrail above, so name them rather
                    # than letting the totals look authoritative.
                    "skipped_repos": skipped_repos,
                }
                if fleet_totals
                else None,
            },
        )

    # Execute the decision
    if decision.action == ScaleAction.UP:
        from .runners import provision_runner

        result = provision_runner(
            gh,
            config.runner_scaling,
            state.busy_runners,
            dry_run=False,
        )
        if result.ok:
            # Record scale event
            from .runners import record_scale_event

            record_scale_event(paths.root, "up")
            return CommandResult(
                ok=True,
                message=f"autoscale: scaled up by {decision.count} - {decision.reason}",
                data={
                    "decision": {
                        "action": decision.action.value,
                        "count": decision.count,
                        "reason": decision.reason,
                    },
                    "provisioning": {
                        "runner_name": result.runner_name,
                        "runner_dir": str(result.runner_dir) if result.runner_dir else None,
                    },
                },
            )
        else:
            return CommandResult(
                ok=False,
                message=f"autoscale: scale up failed - {result.error}",
                data={
                    "decision": {
                        "action": decision.action.value,
                        "count": decision.count,
                        "reason": decision.reason,
                    },
                    "error": result.error,
                },
            )
    elif decision.action == ScaleAction.DOWN:
        managed_root = Path(config.runner_scaling.managed_root)
        if not managed_root.exists():
            return CommandResult(
                ok=False,
                message=f"managed_root does not exist: {managed_root}",
                data={},
            )

        removed_count, errors = scale_down_idle_runners(
            managed_root,
            config.runner_scaling.runner_dir_prefix,
            gh,
            config.runner_scaling,
            paths.root,
            dry_run=False,
        )
        return CommandResult(
            ok=removed_count > 0 or not errors,
            message=f"autoscale: scaled down by {removed_count} - {decision.reason}",
            data={
                "decision": {
                    "action": decision.action.value,
                    "count": decision.count,
                    "reason": decision.reason,
                },
                "removed_count": removed_count,
                "errors": errors,
            },
        )
    else:
        # No action
        return CommandResult(
            ok=True,
            message=f"autoscale: no action - {decision.reason}",
            data={
                "decision": {
                    "action": decision.action.value,
                    "count": decision.count,
                    "reason": decision.reason,
                },
            },
        )


def run_runners_allocate(args: argparse.Namespace) -> CommandResult:
    """Rebalance running runner listeners across repos by live queue demand.

    Unlike ``runners autoscale``, which grows or shrinks *this* repo's pool,
    this command is host-wide: it discovers every repo with runners registered
    under ``managed_root`` and redistributes one shared budget of running
    listeners between them. Registration is never touched — slots move by
    starting and stopping already-configured listeners.

    Returns an error if the runner_allocation feature is not enabled.
    """
    repo_root = find_repo_root(args.repo, explicit=args.repo is not None)
    # runner_allocation is a fleet-wide knob declared in the global fleet config
    # layer. An unreachable global layer silently flips it to its dataclass
    # default (enabled=False), and this command would then report "not enabled"
    # -- the exact #623 silent-disable failure. Require the global layer and
    # fail loudly instead, so an unready volume is distinguishable from a
    # fleet that genuinely opted out of runner allocation.
    try:
        config = load_layered_config(
            repo_root,
            args.config,
            fleet_dir_override=args.fleet_dir,
            require_global=True,
        )
    except (ConfigError, RepoNotFoundError) as exc:
        return CommandResult(
            ok=False,
            message=f"config load failed, cannot decide runner_allocation: {exc}",
            data={},
        )
    paths = runtime_paths(repo_root, config.runtime.state_dir)

    if not config.runner_allocation.enabled:
        return CommandResult(
            ok=False,
            message="runner_allocation feature is not enabled in config",
            data={},
        )

    dry_run = getattr(args, "dry_run", False)
    gh = GitHub(repo_root=repo_root, runtime=config.runtime, dry_run=dry_run)

    result = run_allocation_pass(
        gh,
        config.runner_allocation,
        managed_root_fallback=config.runner_scaling.managed_root,
        fleet_dir_override=args.fleet_dir,
        state_path=paths.state_file,
        dry_run=dry_run,
        source=CLI_ALLOCATION_SOURCE,
        full_pass_interval_seconds=config.supervisor.full_pass_interval_seconds,
    )

    if result.error:
        return CommandResult(ok=False, message=f"allocate: {result.error}", data={})

    data: dict[str, Any] = {
        "dry_run": dry_run,
        "notes": list(result.notes),
        "applied": [
            {
                "repo": r.change.repo,
                "runner": r.change.runner_name,
                "action": r.change.action.value,
                "ok": r.ok,
                "message": r.message,
            }
            for r in result.results
        ],
    }
    if result.plan is not None:
        data["plan"] = plan_summary(result.plan)

    if result.skipped:
        return CommandResult(
            ok=True,
            message=f"allocate: no action - {'; '.join(result.notes) or 'nothing to do'}",
            data=data,
        )

    prefix = "would " if dry_run else ""
    return CommandResult(
        ok=result.ok,
        message=(
            f"allocate: {prefix}start {result.started}, {prefix}park {result.parked} "
            f"(budget {result.plan.budget if result.plan else 0})"
        ),
        data=data,
    )


def run_command(app: OrchestratorApp, args: argparse.Namespace) -> CommandResult:
    if args.command == "roll-call":
        return app.status()
    if args.command == "bootstrap-labels":
        return app.bootstrap_labels()
    if args.command == "intake":
        return app.intake()
    if args.command == "claim":
        return app.claim(args.issue, release=args.release)
    if args.command == "mop-up":
        return app.reconcile(fix=args.fix)
    if args.command == "work":
        return app.dispatch(args.limit, only_issues=args.issues)
    if args.command == "review-queue":
        return app.review_queue()
    if args.command == "why-charlie-hate":
        return app.review(args.pr, cross_family=args.cross_family)
    if args.command == "why-charlie-hate-spec":
        try:
            return app.spec_review(args.spec_file)
        except OSError as exc:
            return CommandResult(False, f"OS error: {exc}", {})
    if args.command == "verdict":
        try:
            return app.record_review(
                args.pr,
                args.decision,
                summary=args.summary,
                summary_file=args.summary_file,
                comment=args.comment,
                reviewed_head=args.reviewed_head,
            )
        except OSError as exc:
            return CommandResult(False, f"OS error: {exc}", {})
    if args.command == "unescalate":
        try:
            return app.unescalate(args.pr, args.issue, dry_run=args.dry_run)
        except OSError as exc:
            return CommandResult(False, f"OS error: {exc}", {})
    if args.command == "ship-it":
        return app.merge_ready(args.pr, merge=args.merge)
    if args.command == "bash-rats":
        from .supervise import run_supervised, try_acquire_supervisor_lock

        if args.once:
            # Single-pass mode: check the supervisor lock first to avoid double-
            # dispatching through the governor read→launch window when a supervised
            # loop is already running.
            lock_path = app.paths.root / "supervisor.lock"
            lock = try_acquire_supervisor_lock(lock_path)
            if lock is None:
                return CommandResult(
                    False,
                    "supervisor already running (supervisor.lock held)",
                    {},
                )
            try:
                return app.loop(args.limit, merge=args.merge)
            finally:
                lock.release()
        return run_supervised(
            app,
            limit=args.limit,
            merge=args.merge,
            poll_interval_override=args.poll_interval,
            max_runtime_override=args.max_runtime,
        )
    return CommandResult(False, f"unknown command: {args.command}", {})


def _render_api_worker_report(data: dict[str, Any]) -> None:
    """Print the api-worker fleet report line when present (issue #483).

    The line is omitted entirely when no registered repo configures the
    ``api_worker`` section (``api_worker_report`` is ``None`` in that case).
    """
    report = data.get("api_worker_report")
    if isinstance(report, dict) and report.get("line"):
        print(f"  {report['line']}")


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

    configure_logging(verbose=getattr(args, "verbose", False))
    try:
        if args.command == "doctor":
            result = run_doctor_command(args)
        elif args.command == "fleet":
            if args.fleet_command == "status":
                result = run_fleet_status(args)
            elif args.fleet_command == "review-queue":
                result = run_fleet_review_queue(args)
            elif args.fleet_command == "work":
                result = run_fleet_work(args)
            elif args.fleet_command == "bash-rats":
                result = run_fleet_bash_rats(args)
            elif args.fleet_command == "supervise":
                result = run_fleet_supervise_command(args)
            else:
                result = CommandResult(False, f"unknown fleet command: {args.fleet_command}", {})
        elif args.command == "runners":
            if args.runners_command == "status":
                result = run_runners_status(args)
            elif args.runners_command == "ensure-started":
                result = run_runners_ensure_started(args)
            elif args.runners_command == "scale-down":
                result = run_runners_scale_down(args)
            elif args.runners_command == "autoscale":
                result = run_runners_autoscale(args)
            elif args.runners_command == "allocate":
                result = run_runners_allocate(args)
            else:
                result = CommandResult(
                    False, f"unknown runners command: {args.runners_command}", {}
                )
        elif args.command == "worktree-clean":
            result = run_worktree_clean_command(args)
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
    except StateLockBusy as exc:
        print(f"state lock busy: {exc}", file=sys.stderr)
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
            _render_api_worker_report(result.data)
        elif args.fleet_command in ("work", "bash-rats"):
            repos = result.data.get("repos", {})
            for repo_key, repo_data in sorted(repos.items()):
                # repo_data now includes the ok field from fleet_dispatch
                ok = repo_data.get("ok", True)
                status = "OK" if ok else "FAILED"
                if repo_data.get("skipped"):
                    status = "SKIPPED"
                print(f"  {repo_key}: {status}")
                if status != "OK" and repo_data.get("message"):
                    print(f"    {repo_data['message']}")
            digest = result.data.get("digest", {})
            event_count = digest.get("count", 0)
            orphan_sweep_calls = digest.get("orphan_sweep_calls", 0)
            print(
                f"  Digest: {event_count} attention event(s), {orphan_sweep_calls} orphan sweep call(s)"
            )
            _render_api_worker_report(result.data)
    elif args.command == "runners" and not json_output:
        print(result.message)
        if args.runners_command == "status" and result.ok:
            pool_size = result.data.get("pool_size", {})
            queue_depth = result.data.get("queue_depth", {})
            host_headroom = result.data.get("host_headroom", {})
            pressure = result.data.get("pressure", "unknown")
            print(
                f"  Pool: {pool_size.get('total', 0)} total, {pool_size.get('online', 0)} online, "
                f"{pool_size.get('busy', 0)} busy, {pool_size.get('idle', 0)} idle"
            )
            print(
                f"  Queue: {queue_depth.get('queued', 0)} queued, {queue_depth.get('in_progress', 0)} in progress"
            )
            print(
                f"  Host: {host_headroom.get('free_ram_gb', 0)} GB free RAM, {host_headroom.get('cpu_percent', 0)}% CPU"
            )
            print(f"  Pressure: {pressure}")
        elif args.runners_command == "ensure-started" and result.ok:
            started_count = result.data.get("started_count", 0)
            messages = result.data.get("messages", [])
            print(f"  Started: {started_count} runner(s)")
            for message in messages:
                print(f"  {message}")
        elif args.runners_command == "scale-down":
            removed_count = result.data.get("removed_count", 0)
            errors = result.data.get("errors", [])
            print(f"  Removed: {removed_count} runner(s)")
            if errors:
                print("  Errors:")
                for error in errors:
                    print(f"    {error}")
        elif args.runners_command == "autoscale":
            decision = result.data.get("decision", {})
            action = decision.get("action", "unknown")
            count = decision.get("count", 0)
            reason = decision.get("reason", "")
            print(f"  Decision: {action}({count}) - {reason}")
            if action == "up" and result.ok:
                provisioning = result.data.get("provisioning", {})
                runner_name = provisioning.get("runner_name")
                runner_dir = provisioning.get("runner_dir")
                if runner_name:
                    print(f"  Provisioned: {runner_name}")
                if runner_dir:
                    print(f"  Directory: {runner_dir}")
            elif action == "down":
                removed_count = result.data.get("removed_count", 0)
                errors = result.data.get("errors", [])
                print(f"  Removed: {removed_count} runner(s)")
                if errors:
                    print("  Errors:")
                    for error in errors:
                        print(f"    {error}")
    else:
        print_result(result, json_output=args.json_output)

    return 0 if result.ok else 1
