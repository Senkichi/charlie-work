from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from . import CLI_NAME
from .closing_keyword_gate import find_unexpected_closing_references
from .config import ConfigError, find_config_path
from .doctor import DoctorCheck, run_doctor
from .fleet_dispatch import (
    compute_api_worker_fleet_report,
    fleet_loop,
    run_fleet_supervise,
    run_fleet_supervise_loop,
)
from .supervise_loop import DEFAULT_MAX_RELAUNCHES, EXIT_RESTART_REQUESTED
from .fleet_paths import fleet_dir
from .fleet_registry import _load_registry, touch_repo, count_fleet_runners
from .global_config import load_layered_config
from .github import (
    CLOSING_KEYWORD_PR_FIELDS,
    GitHub,
    GitHubError,
    defang_closing_keywords,
    linked_issue_number,
)
from . import layout
from .logging_setup import configure_logging
from .instrumentation import query_events
from .notify import AttentionDigest, AttentionEntry, emit_digest
from .paths import RepoNotFoundError, find_repo_root, resolved_layout, runtime_paths
from .quiesce import check_quiescence
from .state import StateLockBusy, load_state_locked, utc_now
from .state_migration import apply_state_dir_migration, gather_migration_inputs
from .supervise import orchestrator_root, self_deploy
from ci_fleet.charlie_work_adapter import (
    CLI_ALLOCATION_SOURCE,
    UNATTENDED_ALLOCATION_SOURCE,
    FleetTotals,
    ScaleAction,
    decide_autoscale,
    ensure_runners_started,
    format_runner_pool_state,
    is_in_cooldown,
    is_pool_idle_for_minutes,
    observe_runner_pool,
    plan_summary,
    run_allocation_pass,
    scale_down_idle_runners,
)

# Not part of the charlie_work_adapter migration surface (issue #909's
# reporter is a new consumer, not one of the four already-migrated ones), so
# NOTE: ci_fleet's shadow/rollback cluster (diff_journal, shadow_gate) is
# deliberately NOT imported here. Those modules exist only to serve
# `charlie runners shadow-status`, and ci_fleet is retiring them as the
# legacy-planner rollback path closes. A third member, shadow_pass, has
# already been deleted upstream -- the retirement is in progress, not
# hypothetical. Imported at module scope, the deletion
# of any one of them takes down the entire CLI -- including
# `charlie runners allocate` and `fleet supervise`, whose entry points import
# this module. That is not hypothetical: it happened on main (issue #929),
# and CI cannot see it, because ci_fleet is an editable path dependency whose
# working tree is live here while CI resolves the committed tree.
#
# They are imported inside run_runners_shadow_status instead, so a retired
# module breaks exactly the one read-only command that reports on it.
# There is no rule against a direct ci_fleet import (only against ci_fleet
# importing back out through the adapter -- see that module's docstring on
# the one-way boundary); the confinement here is about blast radius, not
# layering.
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

    merge_check = subparsers.add_parser(
        "merge-check",
        help=(
            "Preflight: exit 0 only if PR is approved at its current head "
            "(issue #894). Read-only — never merges. Intended for a PreToolUse "
            "hook to gate a raw `gh pr merge`, which otherwise bypasses every "
            "authorization check in this codebase."
        ),
    )
    merge_check.add_argument("pr", type=int, help="PR number to check")

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

    fleet_supervise_loop = fleet_sub.add_parser(
        "supervise-loop",
        help=(
            "Run `fleet supervise`, relaunching immediately when it exits to pick up "
            "new code, bounded by --max-relaunches (#862)."
        ),
    )
    fleet_supervise_loop.add_argument(
        "--max-relaunches",
        type=int,
        default=DEFAULT_MAX_RELAUNCHES,
        dest="max_relaunches",
        help=(
            "Maximum immediate relaunches before exiting and letting the scheduled "
            f"task's next tick take over (default {DEFAULT_MAX_RELAUNCHES})."
        ),
    )
    fleet_supervise_loop.add_argument(
        "supervise_args",
        nargs="*",
        help=(
            "Arguments forwarded verbatim to `fleet supervise`, e.g. "
            "`supervise-loop -- --max-runtime 0`. Passthrough rather than "
            "re-declared here so a new supervise flag needs no change in this "
            "wrapper."
        ),
    )

    runners = subparsers.add_parser("runners")
    runners_sub = runners.add_subparsers(dest="runners_command", required=True)
    runners_sub.add_parser("status")
    runners_sub.add_parser(
        "shadow-status",
        help=(
            "Read-only report of the ci_fleet shadow-planner state: the "
            "actuating planner as of the last source='prologue' runner_allocation "
            "event, a newer source='cli' row if one is configured-but-not-yet-in-"
            "effect, and the shadow-vs-live agreement streak from the diff "
            "journal. Resolves both source locations itself (issue #909); no "
            "arguments to point them elsewhere."
        ),
    )
    ensure_started_parser = runners_sub.add_parser("ensure-started")
    _add_dry_run(ensure_started_parser)
    ensure_started_parser.add_argument(
        "--force",
        action="store_true",
        dest="force",
        help=(
            "Bypass the runner_allocation single-controller guard. By default "
            "ensure-started refuses to run when runner_allocation.enabled is true, "
            "because relaunching every not-running listener silently undoes slot "
            "parking and burns a full demand_idle_samples hysteresis window "
            "reconverging. Use this only for a deliberate manual recovery that "
            "runners allocate cannot do; otherwise prefer `charlie runners allocate`."
        ),
    )
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

    worktree_clean_parser = subparsers.add_parser("worktree-clean")
    _add_dry_run(worktree_clean_parser)

    closing_keyword_check = subparsers.add_parser(
        "closing-keyword-check",
        help=(
            "CI gate (issue #790): fail if the PR body or any commit message "
            "contains an unnegated closing keyword (Closes/Fixes/Resolves #N) "
            "referencing an issue other than this PR's own declared target. "
            "GitHub's native auto-close-on-merge scans both surfaces with no "
            "negation awareness at all; this is a required PR check, not a "
            "label-transition helper."
        ),
    )
    closing_keyword_check.add_argument("--pr", type=int, required=True)

    migrate_parser = subparsers.add_parser(
        "migrate-state-dir",
        help="Plan (and optionally apply) a move of a legacy state dir to its new root",
    )
    # --src/--dst are optional overrides. Their defaults are derived, never a
    # repeated literal: src is wherever this repo's config currently points
    # runtime.state_dir at (the "legacy" location for the duration of the
    # migration window), dst is the package-wide default state root from
    # layout.py. Overrides exist so an operator can rehearse against a copied
    # tree instead of the live one.
    migrate_parser.add_argument(
        "--src",
        default=None,
        help="Legacy state dir to move from (default: this repo's configured runtime.state_dir)",
    )
    migrate_parser.add_argument(
        "--dst",
        default=None,
        help="New state dir to move into (default: the package's default state root)",
    )
    migrate_parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move. Without this the command only plans and prints.",
    )
    migrate_parser.add_argument(
        "--quiesce-pattern",
        action="append",
        dest="quiesce_patterns",
        default=None,
        metavar="REGEX",
        help=(
            "Regex matched against live process command lines to prove the fleet is "
            "stopped before --apply acts; repeatable. With --apply and no pattern given, "
            "quiescence cannot be established and the move is refused. Without --apply, "
            "patterns (if given) are only reported informationally."
        ),
    )
    _add_dry_run(migrate_parser)

    tripwire = subparsers.add_parser(
        "tripwire",
        help="Manage the #502 post-merge unauthorized-merge tripwire",
    )
    tripwire_sub = tripwire.add_subparsers(dest="tripwire_command", required=True)
    tripwire_sub.add_parser(
        "status",
        help=(
            "Show pending unauthorized-merge findings (detected and not yet "
            "acknowledged) with the PR, branch and decision that pinned "
            "ok=False (issue #933). Reads state.json only — no gh calls, and "
            "it never arms the baseline as a side effect."
        ),
    )
    tripwire_ack = tripwire_sub.add_parser(
        "ack",
        help=(
            "Acknowledge a post-arming unauthorized-merge finding so it stops "
            "pinning ok=False on every pass (issue #673). Requires an explicit "
            "--reason; the tripwire never auto-acknowledges."
        ),
    )
    tripwire_ack.add_argument("pr", type=int, help="PR number to acknowledge")
    tripwire_ack.add_argument(
        "--reason",
        default=None,
        help=(
            "Why this finding is triaged (e.g. 'root cause fixed in #N', "
            "'confirmed benign per #634 audit'). Mandatory — a tripwire that "
            "can be silenced silently is no control. Validated by the handler "
            "so a missing reason exits 1 (a command failure) rather than 2 "
            "(an argparse usage error), matching every other command."
        ),
    )
    tripwire_ack.add_argument(
        "--by",
        default=None,
        help="Operator who acknowledged the finding (recorded for audit).",
    )

    return parser


def _assert_config_repo_matches(config_arg: Path | None, repo_root: Path) -> None:
    """Refuse to apply one repo's ``--config`` to another repo's state (issue #895).

    ``--config`` selects the *config*; it never selected the *state*. ``repo_root``
    comes from ``--repo`` (defaulting to cwd), and ``runtime_paths`` resolves a
    **relative** ``state_dir`` against it. Every managed repo uses a relative
    ``state_dir``, so ``charlie --config <job-cannon> tripwire ack 1392`` run from
    a charlie-work cwd loaded job-cannon's settings and wrote job-cannon's ack into
    *charlie-work's* state file — exit 0, no warning, and job-cannon's finding
    still pinned ``ok=False``.

    That silence is the danger, not the misroute: a misdirected write into a keyed
    map is indistinguishable from a legitimate one afterwards. charlie-work's ack
    map ended up holding entries for PR #1392/#1408 while its own numbering was in
    the 800s and climbing — a security control pre-disarmed for two PRs that did
    not exist yet.

    Fails closed with the corrective invocation rather than inferring ``repo_root``
    from the config's location: a shared or layered config may legitimately live
    outside the repo it configures, and silently re-rooting would trade this bug
    for a subtler one. Only fires when the config provably belongs to a *different
    git work tree* — a config outside any repo is left alone for exactly that reason.
    """
    if config_arg is None:
        return
    config_dir = config_arg.expanduser().resolve().parent
    if not config_dir.exists():
        return
    config_repo = find_repo_root(config_dir, explicit=False)
    if config_repo == repo_root or not (config_repo / ".git").exists():
        return
    raise ConfigError(
        f"--config points into {config_repo}, but state resolves against {repo_root}. "
        f"--config does not select the state directory. "
        f"Pass --repo {config_repo} to operate on that repo."
    )


def build_app(args: argparse.Namespace) -> OrchestratorApp:
    repo_root = find_repo_root(args.repo, explicit=args.repo is not None)
    _assert_config_repo_matches(args.config, repo_root)
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
    # #6-G: doctor exists to diagnose a broken repo, so it must not itself
    # crash on the exact condition it is meant to diagnose (e.g. the
    # unknown-config-key ConfigError that silently killed the cw fleet lane
    # three times on 2026-07-29). Render the parse failure as a structured
    # finding instead of letting it propagate to main()'s generic
    # `except (ConfigError, ValueError)` handler, which prints to stderr and
    # exits 2 with no machine-readable finding at all.
    try:
        config = load_layered_config(repo_root, args.config, fleet_dir_override=args.fleet_dir)
    except ConfigError as exc:
        check = DoctorCheck(
            name="config file",
            ok=False,
            detail=f"{config_path or 'orchestrator.config.yaml'}: {exc}",
        )
        return CommandResult(
            False,
            "doctor: 1 finding(s), at least one blocking",
            {"checks": [check.to_dict()]},
        )
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
        resolved_layout(config, repo_root).worktrees,
        state,
        config,
        gh,
        dry_run=args.dry_run,
    )
    return CommandResult(result.ok, result.message, result.data)


def run_closing_keyword_check_command(args: argparse.Namespace) -> CommandResult:
    """CI gate (issue #790): fail on any unnegated closing keyword pointing off-target.

    Fetches the PR's title/body/branch (`GitHub.pr_view`, deliberately scoped
    to `CLOSING_KEYWORD_PR_FIELDS` rather than the general-purpose
    `PR_VIEW_FIELDS` — this gate never touches CI/review/label state, and
    `PR_VIEW_FIELDS`'s `statusCheckRollup` triggers a nested GraphQL
    connection the default Actions `GITHUB_TOKEN` cannot read without
    additional scope grants; see `CLOSING_KEYWORD_PR_FIELDS`'s docstring for
    the two live failures this caused) and every commit's raw message
    (`GitHub.pr_commits` — the REST endpoint, not `gh pr view --json commits`, whose GraphQL fields
    truncate/corrupt long commit messages; see `GitHub.pr_commits`'s
    docstring). The PR's own declared target issue is resolved the same way
    charlie-work's own label-transition binding resolves it
    (`linked_issue_number`: same-repo branch-prefix first, then an unnegated
    closing keyword in the PR's own title/body) — that single number is the
    only exemption `find_unexpected_closing_references` allows. Everything
    else it finds is a reference GitHub's native auto-close-on-merge will act
    on regardless of what this codebase intends, because that GitHub feature
    scans PR body + every commit message with no negation awareness (issue
    #790; PR #788's own commit text is the regression fixture proving this).
    """
    repo_root = find_repo_root(args.repo, explicit=args.repo is not None)
    config = load_layered_config(repo_root, args.config, fleet_dir_override=args.fleet_dir)
    gh = GitHub(repo_root=repo_root, runtime=config.runtime, dry_run=args.dry_run)

    pr = gh.pr_view(args.pr, fields=CLOSING_KEYWORD_PR_FIELDS)
    if not pr:
        return CommandResult(False, f"closing-keyword-check: could not fetch PR #{args.pr}", {})

    commits = gh.pr_commits(args.pr)
    if commits is None:
        return CommandResult(
            False, f"closing-keyword-check: could not fetch commits for PR #{args.pr}", {}
        )
    commit_messages = [str((c.get("commit") or {}).get("message") or "") for c in commits]

    intended = linked_issue_number(
        pr,
        is_cross_repository=pr.get("isCrossRepository"),
        branch_prefix=config.dispatch.branch_prefix,
    )

    findings = find_unexpected_closing_references(
        pr_body=str(pr.get("body") or ""),
        commit_messages=commit_messages,
        intended_issue_number=intended,
    )

    data = {
        "pr": args.pr,
        "intended_issue_number": intended,
        "findings": [
            {
                "issue_number": finding.issue_number,
                "source": finding.source,
                "matched_text": finding.matched_text,
            }
            for finding in findings
        ],
    }

    if findings:
        lines = [
            f"  issue #{finding.issue_number} via {finding.source}: "
            f"{finding.matched_text!r} -> reword to {defang_closing_keywords(finding.matched_text)!r}"
            for finding in findings
        ]
        message = (
            f"closing-keyword-check: {len(findings)} unexpected closing reference(s) on "
            f"PR #{args.pr} (declared target: "
            f"{'#' + str(intended) if intended is not None else 'none resolved'})\n"
            + "\n".join(lines)
            + "\nGitHub will auto-close these issues on merge unless the wording above is "
            "changed to the suggested rewrite (or the reference is dropped entirely)."
        )
        return CommandResult(False, message, data)

    return CommandResult(
        True,
        f"closing-keyword-check: clean (PR #{args.pr}, declared target: "
        f"{'#' + str(intended) if intended is not None else 'none resolved'})",
        data,
    )


def _resolve_migration_root(raw: str, repo_root: Path) -> Path:
    """Resolve a ``--src``/``--dst`` argument against *repo_root* when relative."""
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate


def _migration_path_key(path: Path) -> str:
    """Fold case/separator so a resolved src/dst pair compares equal correctly.

    Mirrors the normalization discipline ``state_migration._normalize_path_key``
    applies for the same hazard (git-porcelain forward slashes vs. disk
    backslashes, drive-letter case) without importing a private name across
    that module's boundary.
    """
    return os.path.normcase(os.path.normpath(str(path)))


def _render_migration_plan(plan) -> str:
    """Human-readable rendering of a plan: counts, then every blocked child.

    Blocked children are printed with their reasons *and* remediation, because the
    operator's next action is always "clear the blockers, re-plan" -- a bare count
    would send them back to the filesystem to work out why.
    """
    lines = [
        f"src: {plan.src_root}",
        f"dst: {plan.dst_root}",
        f"children: {len(plan.children)}  movable: {len(plan.movable)}  "
        f"blocked: {len(plan.blocked)}",
    ]
    for child in plan.blocked:
        lines.append(f"  BLOCKED {child.name}")
        for reason in child.reasons:
            lines.append(f"      reason: {reason}")
        for step in child.remediation:
            lines.append(f"      remediate: {step}")
    return "\n".join(lines)


def run_migrate_state_dir_command(
    args: argparse.Namespace,
    *,
    planner=gather_migration_inputs,
    actuator=apply_state_dir_migration,
    quiescence_checker=check_quiescence,
) -> CommandResult:
    """Plan, and with ``--apply``, actuate a legacy state-dir move.

    Plan-only is the default; ``--apply`` is the explicit opt-in. A global or
    subcommand ``--dry-run`` *overrides* ``--apply`` rather than the other way
    round, so the two flags together can only ever be safe: the failure mode of
    the opposite precedence is an operator who wrote ``--dry-run`` watching a
    real migration run.

    ``--src``/``--dst`` default to the repo's *currently configured*
    ``runtime.state_dir`` and the package's default state root respectively --
    never a literal re-spelled here -- so the common case needs no flags at
    all. If they resolve to the same place there is nothing to migrate.

    Quiescence (proof the fleet is stopped) gates ``--apply`` only. A dry run
    must keep working against a live fleet -- that is how an operator inspects
    a plan before committing -- so it only *reports* quiesce status. Patterns
    are supplied by the caller via ``--quiesce-pattern`` (repeatable); there is
    no built-in default list (CLAUDE.md rule 9: no embedded manual lists), so
    ``--apply`` with none given is refused rather than silently skipping the
    check.
    """
    repo_root = find_repo_root(args.repo, explicit=args.repo is not None)
    config = load_layered_config(repo_root, args.config, fleet_dir_override=args.fleet_dir)

    src_root = (
        _resolve_migration_root(args.src, repo_root)
        if args.src is not None
        else runtime_paths(repo_root, config.runtime.state_dir).root
    )
    dst_root = (
        _resolve_migration_root(args.dst, repo_root)
        if args.dst is not None
        # ``.resolve()`` here (not just in ``layout.default_state_root``) matters:
        # ``runtime_paths`` above resolves symlinks/junctions in the *whole* src
        # path, not just ``repo_root``. Without a matching resolve on this side,
        # a repo whose ``.var`` is itself a symlink/junction would make the two
        # roots compare unequal even when they name the same on-disk location,
        # producing a same-place migration plan instead of the intended
        # already-migrated short-circuit. Safe on a not-yet-created dst: Path
        # .resolve() does not require the path to exist.
        else layout.default_state_root(repo_root).resolve()
    )

    if _migration_path_key(src_root) == _migration_path_key(dst_root):
        return CommandResult(
            True,
            f"already migrated: src and dst both resolve to {dst_root}",
            {
                "src_root": str(src_root),
                "dst_root": str(dst_root),
                "already_migrated": True,
                "applied": False,
            },
        )

    patterns = tuple(args.quiesce_patterns) if args.quiesce_patterns else ()

    plan = planner(repo_root=repo_root, src_root=src_root, dst_root=dst_root)
    rendered = _render_migration_plan(plan)
    data = {
        "src_root": str(plan.src_root),
        "dst_root": str(plan.dst_root),
        "children": len(plan.children),
        "movable": len(plan.movable),
        "blocked": [child.name for child in plan.blocked],
        "applied": False,
    }

    if not plan.ok:
        return CommandResult(False, f"{rendered}\nplan failed: {plan.error}", data)

    acting = args.apply and not args.dry_run

    if not acting:
        if args.apply and args.dry_run:
            note = "(dry-run: --apply ignored, nothing moved)"
        else:
            note = "(plan only; pass --apply to move)"
        if patterns:
            report = quiescence_checker(patterns=patterns)
            note = f"{note}\nquiesce (informational, not enforced for a dry run): {report.summary}"
        else:
            note = f"{note}\nquiesce: not checked (no --quiesce-pattern given)"
        return CommandResult(True, f"{rendered}\n{note}", data)

    if not patterns:
        return CommandResult(
            False,
            f"{rendered}\nrefusing to apply: no --quiesce-pattern given, "
            "quiescence cannot be established",
            data,
        )

    report = quiescence_checker(patterns=patterns)
    if not report.ok:
        return CommandResult(
            False,
            f"{rendered}\nrefusing to apply: fleet is not quiescent\n{report.summary}",
            data,
        )

    outcome = actuator(plan)
    data = {**data, "applied": outcome.ok, "moved": list(outcome.moved)}
    if not outcome.ok:
        data = {**data, "aborted_at": outcome.aborted_at}
        return CommandResult(
            False,
            f"{rendered}\nmigration failed after {len(outcome.moved)} moved: {outcome.error}",
            data,
        )
    return CommandResult(True, f"{rendered}\nmoved {len(outcome.moved)} children", data)


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
    fleet_json_path = layout.fleet_registry_path()
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
    fleet_json_path = layout.fleet_registry_path()
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


def run_runners_shadow_status(args: argparse.Namespace) -> CommandResult:
    """Report the ci_fleet shadow-planner rollback state (issue #909).

    Read-only. Reads two stores that live in two different directories and
    resolves both itself so neither is an operator input -- the "obvious"
    inference that the events DB lives beside the journal is wrong and fails
    silently (issue #909, trap 1):

    - ``events.db`` (this repo's state dir, ``<repo>/.var/charlie-work/``):
      ``runner_allocation`` events. ``source`` and ``actuating_planner`` live
      inside the JSON ``payload`` column, not as SQL columns, so they are read
      back out of the parsed dicts ``query_events`` already returns rather
      than pushed into a ``json_extract`` WHERE clause.
    - ``shadow-planner-diff.jsonl`` (the global fleet dir, ``fleet_dir()``):
      the shadow-vs-live diff journal.

    ``source='prologue'`` (the supervisor) is the only row that reflects what
    is actually actuating. The supervisor loads config once at startup and
    reuses the same object every loop iteration (see
    ``runner_allocation_pass.py``'s "the single flip point" comment), so after
    a config flip there can be a newer ``source='cli'`` row naming the *new*
    planner while the supervisor is still actuating the *old* one until it is
    replaced. That makes the filter load-bearing, not decorative (issue #909,
    trap 2): this command has no flag to drop it, and surfaces a newer
    ``cli`` row separately, explicitly labelled "configured, not yet in
    effect" rather than folding it into "actuating" where it would read as
    confirmation.

    The journal itself is NOT filtered by ``source`` -- unlike the events-db
    rows, ``source`` was added to :class:`ci_fleet.diff_journal.DiffRecord`
    recently and is ``None`` on the large majority of existing records
    (verified 2026-08-04: 1024/1041 have no ``source`` at all). Filtering the
    journal by it would silently drop nearly the whole corpus.

    Two agreement streaks are reported, not one, because the obvious single
    number is hollow: the fleet is idle almost every pass, so a streak over
    *all* passes is dominated by two planners agreeing to do nothing.
    Verified 2026-08-04: of 1041 journal records, only 1 has a non-empty
    ``live_plan["changes"]`` (a single park action) -- the change-emitting
    path has been compared exactly once, even though the all-passes streak is
    over a thousand. Reporting only the all-passes number would let an
    operator read "agreed 1040 times" as "the acting path is thoroughly
    validated," which it is not. The change-restricted streak is the
    load-bearing one and is labelled as such in both the data and the
    rendered output.

    The gate verdict (``ci_fleet.shadow_gate.evaluate``) is real §6.3
    production logic, not reimplemented here -- its ``streak``/``total`` match
    this command's "all passes" figures exactly, since both walk the same
    journal the same way. Per the issue's closing note: ``ok: True`` here
    licenses "agreed N consecutive times" and nothing stronger. In
    particular ``adjudication_ok`` is true *vacuously* whenever there have
    been zero disagreements to adjudicate -- it does not mean the adjudication
    path has ever been exercised, and this command does not claim it does.
    """
    # Imported here, not at module scope, on purpose -- see the note by the
    # ci_fleet imports at the top of this file. This command is the only
    # consumer of ci_fleet's shadow/rollback cluster, and that cluster is
    # being retired; confining the import confines the failure to this
    # command instead of taking down every other `charlie` subcommand.
    # Deliberately NOT wrapped in try/except ImportError: a retired module
    # should fail loudly here, where the traceback names the missing module,
    # rather than being bound to None and surfacing later as an
    # AttributeError with the cause erased.
    from ci_fleet.diff_journal import (
        journal_path as shadow_journal_path,
        read_all as read_shadow_journal,
    )
    from ci_fleet.shadow_gate import (
        REQUIRED_CALENDAR_DAYS,
        REQUIRED_STREAK,
        evaluate as evaluate_shadow_gate,
    )

    repo_root = find_repo_root(args.repo, explicit=args.repo is not None)
    config = load_layered_config(repo_root, args.config, fleet_dir_override=args.fleet_dir)
    paths = runtime_paths(repo_root, config.runtime.state_dir)

    data: dict[str, Any] = {}

    # --- Store 1: events.db (this repo's state dir) -----------------------
    # Same location instrumentation.py's _db_path derives: state.json's
    # parent, i.e. paths.root. Checked for existence before doing anything
    # else so a missing DB is reported as missing rather than silently
    # created -- sqlite3.connect() would otherwise create an empty file the
    # instant query_events() opened it, which is not a "read-only reporter"
    # for a store that was never written.
    events_db_path = paths.root / "events.db"
    events_db_found = events_db_path.exists()
    data["events_db"] = {"path": str(events_db_path), "found": events_db_found}

    prologue_event: dict[str, Any] | None = None
    cli_event: dict[str, Any] | None = None
    # True when the cli row is more recent than the prologue row (or there is
    # no prologue row at all). Decided structurally from insertion order
    # (query_events' id-ascending ordering, reversed below to scan newest
    # first) rather than by comparing the ``ts`` strings: wall-clock time is
    # exactly the kind of signal that can go backwards (NTP step, DST, a
    # rebooted host with a wrong clock), and this comparison is the one
    # trap-2 depends on to keep a dry-run pre-check from reading as
    # confirmation -- it must not be the one field allowed to lie.
    cli_newer_than_prologue = False
    if events_db_found:
        # kind is the only indexed column this query needs; source and
        # actuating_planner are read back out of the parsed payload dict
        # query_events() already returns, per trap 1 above. No limit: the
        # live corpus is ~2000 runner_allocation rows, trivial to scan, and a
        # limit risks missing the latest prologue row behind a burst of cli
        # rows within the window.
        allocation_events = query_events(paths.state_file, kind="runner_allocation")
        for event in reversed(allocation_events):  # most recent (highest id) first
            source = event.get("payload", {}).get("source")
            if source == UNATTENDED_ALLOCATION_SOURCE and prologue_event is None:
                prologue_event = event
            elif source == CLI_ALLOCATION_SOURCE and cli_event is None:
                cli_event = event
                # If no prologue row has been seen yet while scanning newest
                # first, none exists more recent than this cli row.
                cli_newer_than_prologue = prologue_event is None
            if prologue_event is not None and cli_event is not None:
                break

    if prologue_event is not None:
        data["actuating"] = {
            "planner": prologue_event["payload"].get("actuating_planner"),
            "ts": prologue_event.get("ts"),
            "source": UNATTENDED_ALLOCATION_SOURCE,
        }
    else:
        data["actuating"] = None

    # The cli row is surfaced only when it is more recent than the prologue
    # row -- an *older* cli row is just a historical pre-check, not a
    # pending, not-yet-actuated change. Trap 2: this check, and the fact that
    # "actuating" above is keyed on source='prologue' alone, is the entire
    # reason a dry-run pre-check cannot be misread as confirmation.
    if cli_event is not None and cli_newer_than_prologue:
        data["configured_not_yet_in_effect"] = {
            "planner": cli_event["payload"].get("actuating_planner"),
            "ts": cli_event.get("ts"),
            "source": CLI_ALLOCATION_SOURCE,
        }
    else:
        data["configured_not_yet_in_effect"] = None

    # --- Store 2: shadow-planner-diff.jsonl (global fleet dir) -------------
    journal_file = shadow_journal_path(fleet_dir(override=args.fleet_dir))
    journal_found = journal_file.exists()
    data["journal"] = {"path": str(journal_file), "found": journal_found}

    if not journal_found:
        data["agreement_streak"] = None
        data["change_agreement_streak"] = None
        data["gate"] = None
        return CommandResult(ok=True, message="runners shadow-status complete", data=data)

    records = read_shadow_journal(journal_file)

    def _trailing_streak(recs: list[dict[str, Any]]) -> int:
        streak = 0
        for rec in reversed(recs):
            if not rec.get("agreed"):
                break
            streak += 1
        return streak

    data["agreement_streak"] = {"streak": _trailing_streak(records), "total": len(records)}

    # The load-bearing figure (see docstring): restricted to passes whose
    # live plan actually emitted a change, i.e. non-empty
    # live_plan["changes"]. live_plan is a dict, not a list -- len(live_plan)
    # counts its keys ('budget', 'budget_reason', 'changes', 'notes',
    # 'targets') and is always non-zero, which silently reports every no-op
    # pass as a "real" one. The list to check is live_plan["changes"].
    changed_records = [rec for rec in records if (rec.get("live_plan") or {}).get("changes")]
    data["change_agreement_streak"] = {
        "streak": _trailing_streak(changed_records),
        "total": len(changed_records),
    }

    verdict = evaluate_shadow_gate(records)
    data["gate"] = {
        "ok": verdict.ok,
        "streak": verdict.streak,
        "streak_required": REQUIRED_STREAK,
        "streak_ok": verdict.streak_ok,
        "calendar_days": verdict.calendar_days,
        "calendar_days_required": REQUIRED_CALENDAR_DAYS,
        "days_ok": verdict.days_ok,
        "classes_covered": sorted(verdict.classes_covered),
        "classes_missing": sorted(verdict.classes_missing),
        "classes_ok": verdict.classes_ok,
        "unadjudicated": list(verdict.unadjudicated),
        "adjudication_ok": verdict.adjudication_ok,
        "report": verdict.report(),
    }

    return CommandResult(ok=True, message="runners shadow-status complete", data=data)


def run_runners_ensure_started(args: argparse.Namespace) -> CommandResult:
    """Ensure all configured managed runners are running.

    This command:
    - Loads the runner scaling configuration
    - Discovers managed runners under managed_root
    - Relaunches any configured-but-not-running managed runner
    - Returns the number of runners started and status messages

    Returns an error if the runner_scaling feature is not enabled.

    Single-controller guard: when ``runner_allocation.enabled`` is true, this
    command refuses unless ``--force`` is passed. ``ensure_runner_running``
    relaunches any runner where ``not is_runner_launched(...)`` -- exactly the
    state a deliberately parked slot is in -- so running it while allocation
    is enabled restarts every parked listener and silently undoes
    ``runners allocate``'s parking, burning a full ``demand_idle_samples``
    hysteresis window reconverging (CLAUDE.md: "``charlie runners allocate``
    is the only thing allowed to decide which listeners run"). The guard
    enforces that invariant at one boundary rather than documenting it.
    """
    repo_root = find_repo_root(args.repo, explicit=args.repo is not None)
    config = load_layered_config(repo_root, args.config, fleet_dir_override=args.fleet_dir)

    if not config.runner_scaling.enabled:
        return CommandResult(
            ok=False,
            message="runner_scaling feature is not enabled in config",
            data={},
        )

    # Single-controller guard (issue #598): when runner_allocation is enabled,
    # `runners allocate` owns the set of running listeners. ensure-started would
    # relaunch every parked slot (a parked slot is exactly a configured-but-
    # not-running listener), silently undoing allocation and burning a full
    # demand_idle_samples hysteresis window reconverging. Refuse and point at
    # the single controller; --force is the explicit escape hatch for a
    # deliberate manual recovery that allocate cannot do.
    force = getattr(args, "force", False)
    if config.runner_allocation.enabled and not force:
        return CommandResult(
            ok=False,
            message=(
                "runner_allocation is enabled: `charlie runners allocate` owns "
                "which listeners run. ensure-started would relaunch every parked "
                "slot and silently undo allocation. Re-run with --force only for "
                "a deliberate manual recovery that allocate cannot do."
            ),
            data={"runner_allocation_enabled": True},
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
        from ci_fleet.charlie_work_adapter import provision_runner

        result = provision_runner(
            gh,
            config.runner_scaling,
            state.busy_runners,
            dry_run=False,
        )
        if result.ok:
            # Record scale event
            from ci_fleet.charlie_work_adapter import record_scale_event

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
    if args.command == "merge-check":
        return app.merge_check(args.pr)
    if args.command == "ship-it":
        return app.merge_ready(args.pr, merge=args.merge)
    if args.command == "tripwire":
        if args.tripwire_command == "ack":
            return app.ack_unauthorized_merge(args.pr, args.reason or "", by=args.by)
        if args.tripwire_command == "status":
            return app.tripwire_status()
        return CommandResult(False, f"unknown tripwire command: {args.tripwire_command}", {})
    if args.command == "bash-rats":
        from .supervise import run_supervised, try_acquire_supervisor_lock

        if args.once:
            # Single-pass mode: check the supervisor lock first to avoid double-
            # dispatching through the governor read→launch window when a supervised
            # loop is already running.
            lock_path = layout.supervisor_lock_path(app.paths.root)
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
            elif args.fleet_command == "supervise-loop":
                result = run_fleet_supervise_loop(
                    supervise_args=tuple(args.supervise_args),
                    max_relaunches=args.max_relaunches,
                )
            else:
                result = CommandResult(False, f"unknown fleet command: {args.fleet_command}", {})
        elif args.command == "runners":
            if args.runners_command == "status":
                result = run_runners_status(args)
            elif args.runners_command == "shadow-status":
                result = run_runners_shadow_status(args)
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
        elif args.command == "migrate-state-dir":
            result = run_migrate_state_dir_command(args)
        elif args.command == "closing-keyword-check":
            result = run_closing_keyword_check_command(args)
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
        elif args.runners_command == "shadow-status" and result.ok:
            events_db = result.data.get("events_db", {})
            if not events_db.get("found"):
                print(f"  events.db: not found: {events_db.get('path')}")
            else:
                actuating = result.data.get("actuating")
                if actuating is None:
                    print("  Actuating (source=prologue): no runner_allocation event found")
                else:
                    print(
                        f"  Actuating (source=prologue, authoritative): "
                        f"{actuating.get('planner')} as of {actuating.get('ts')}"
                    )
                configured = result.data.get("configured_not_yet_in_effect")
                if configured is not None:
                    print(
                        f"  CONFIGURED, NOT YET IN EFFECT (source=cli, newer): "
                        f"{configured.get('planner')} as of {configured.get('ts')}"
                    )

            journal = result.data.get("journal", {})
            if not journal.get("found"):
                print(f"  journal: not found: {journal.get('path')}")
            else:
                print(f"  Journal: {journal.get('path')}")
                all_streak = result.data.get("agreement_streak") or {}
                change_streak = result.data.get("change_agreement_streak") or {}
                print(
                    f"    Agreement streak (all passes):                     "
                    f"{all_streak.get('streak')}/{all_streak.get('total')}"
                )
                print(
                    f"    Agreement streak (passes with a real change) "
                    f"[LOAD-BEARING]: {change_streak.get('streak')}/{change_streak.get('total')}"
                )
                gate = result.data.get("gate") or {}
                if gate:
                    print(
                        f"  Gate ok={gate.get('ok')} (section 6.3 -- ci_fleet.shadow_gate.evaluate):"
                    )
                    for line in gate.get("report", "").splitlines():
                        print(f"    {line}")
                    # The gate's own streak/total (criterion 1a) is the
                    # all-passes figure above, restated -- printed last here so
                    # it lands as the final word instead of "GATE OPEN". Numbers
                    # only, no editorializing: an operator who reads only the
                    # gate report otherwise ends on an unqualified pass that
                    # buries exactly the gap the load-bearing streak exists to
                    # surface (a change-emitting comparison count of
                    # change_streak.total, not all_streak.total).
                    print(
                        "    Note: criterion 1a's streak counts "
                        f"{all_streak.get('total')} pass(es), of which "
                        f"{(all_streak.get('total') or 0) - (change_streak.get('total') or 0)} "
                        "emitted no change; the change-emitting path has been "
                        f"compared {change_streak.get('total')} time(s)."
                    )
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

    # #862: a supervisor exit that asked to be replaced must be distinguishable
    # from a deliberate stop, so the wrapper relaunches on the former and not
    # the latter. Read out of result.data rather than switched on the command
    # name: main() is generic across every command, and a name list here would
    # need editing for each new long-lived command wanting the same signal.
    # Deliberately NOT gated on result.ok. "Should I be replaced?" is orthogonal
    # to "did I succeed?": a supervisor that self-deployed and then hit an error
    # still has new code on disk and still needs relaunching -- relaunching is
    # the recovery, not a reward for a clean run. Gating this on ok meant a
    # preserved restart signal on a failed result was inert, so the wrapper sat
    # out the interval exactly as in #862.
    if isinstance(result.data, dict) and result.data.get("restart_requested"):
        return EXIT_RESTART_REQUESTED

    return 0 if result.ok else 1
