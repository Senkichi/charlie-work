from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import CLI_NAME
from .closing_keyword_gate import find_unexpected_closing_references
from .mojibake_gate import find_mojibake_in_diff
from .ast_equivalence_gate_command import (
    register_ast_equivalence_check_subparser,
    run_ast_equivalence_check_command,
)
from .private_slug_check_command import (
    register_private_slug_check_subparser,
    run_private_slug_check_command,
)
from .junit_recorded_gate_command import (
    register_junit_recorded_check_subparser,
    run_junit_recorded_check_command,
)
from .config import ConfigError, OrchestratorConfig, find_config_path
from .doctor import DoctorCheck, run_doctor
from .fleet_dispatch import (
    compute_api_worker_fleet_report,
    fleet_loop,
    run_allocation_pass_with_ci_fleet_guard,
    run_fleet_supervise,
    run_fleet_supervise_loop,
)
from .supervise_loop import (
    DEFAULT_MAX_RELAUNCHES,
    EXIT_RESTART_REQUESTED,
    PREFLIGHT_REFUSAL_EXIT_CODE,
)
from .fleet_paths import fleet_dir
from .fleet_registry import _load_registry, touch_repo, count_fleet_runners
from .global_config import load_layered_config
from .github import (
    CLOSING_KEYWORD_PR_FIELDS,
    GitHub,
    GitHubError,
    defang_closing_keywords,
)
from .issue_linking import linked_issue_number
from . import layout
from .dirty_tree import check_working_tree_clean
from .logging_setup import configure_logging
from .instrumentation import query_events
from .notify import AttentionDigest, AttentionEntry, emit_digest
from .paths import RepoNotFoundError, RuntimePaths, find_repo_root, resolved_layout, runtime_paths
from .quiesce import check_quiescence
from .state import StateLockBusy, load_state_locked, utc_now
from .subprocess_runner import run_captured
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


def _add_no_cache_arg(parser: argparse.ArgumentParser) -> None:
    """Add ``--no-cache`` (issue #1463) to ``fleet status`` and ``roll-call``."""
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Bypass the status-snapshot cache (#1463); compute live status.",
    )


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

    roll_call = subparsers.add_parser("roll-call")
    _add_no_cache_arg(roll_call)
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
    subparsers.add_parser("operator-queue")

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

    merge_authorize = subparsers.add_parser(
        "merge-authorize",
        help=(
            "Record an operator's explicit authorization to merge a worker PR "
            "(issue #934). Writes an authorized_override into the PR's "
            "review-decision.json, bound to the current head SHA, so the "
            "tripwire and merge-check read a recorded authorization rather "
            "than inferring one. Requires --reason; never weakens the control."
        ),
    )
    merge_authorize.add_argument("pr", type=int, help="PR number to authorize")
    merge_authorize.add_argument(
        "--reason",
        default=None,
        help=(
            "Why this merge is authorized (e.g. 'CI green, stale decision "
            "overridden after content review'). Mandatory — a tripwire that "
            "can be silenced silently is no control, before the merge as "
            "much as after it."
        ),
    )
    merge_authorize.add_argument(
        "--by",
        default=None,
        help="Operator who authorized the merge (recorded for audit).",
    )
    merge_authorize.add_argument(
        "--sha",
        default=None,
        help=(
            "SHA to bind the authorization to. Defaults to the PR's live "
            "headRefOid. An authorization that does not name the SHA it "
            "authorizes reintroduces the rebase-moved-head hole (#802/#804)."
        ),
    )

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
    fleet_status_parser = fleet_sub.add_parser("status")
    _add_no_cache_arg(fleet_status_parser)
    fleet_sub.add_parser("review-queue")
    fleet_sub.add_parser("operator-queue")

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
    provision_parser = runners_sub.add_parser(
        "provision",
        help=(
            "Manually provision one new runner when the pool is starved, reusing "
            "decide_autoscale()'s guardrails (max_runners, RAM headroom, cooldown). "
            "Scale-up only: never scales down. Issue #826."
        ),
    )
    _add_dry_run(provision_parser)
    provision_parser.add_argument(
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

    mojibake_check = subparsers.add_parser(
        "mojibake-check",
        help=(
            "CI gate (issue #1057): fail if the diff introduces mojibake -- "
            "non-ASCII characters corrupted by a UTF-8/cp1252 round trip "
            "(e.g. em-dashes turned into the a-circumflex/euro/quote sequence). "
            "Scans added lines in the diff against --base (default: origin/main) "
            "using a round-trip detection derived from the encoding process, "
            "not a hardcoded list of bad sequences."
        ),
    )
    mojibake_check.add_argument(
        "--base",
        default="origin/main",
        help="Git ref to diff against (default: origin/main). Uses the "
        "two-dot diff (base..HEAD) which compares trees directly and works "
        "in shallow clones (CI uses fetch-depth: 1) where three-dot "
        "(base...HEAD) cannot resolve the merge-base.",
    )

    register_private_slug_check_subparser(subparsers)
    register_ast_equivalence_check_subparser(subparsers)
    register_junit_recorded_check_subparser(subparsers)

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
    migrate_parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Override the pre-flight clean-tree gate (issue #729): --apply refuses to "
            "run when the tracked working tree differs from HEAD, since the command "
            "executes the working tree while CI only reviewed the committed tree. Pass "
            "this flag only for deliberate local testing on a dirty tree."
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
    ``state_dir``, so ``charlie --config <sibling-repo> tripwire ack 1392`` run from
    a charlie-work cwd loaded the sibling repo's settings and wrote its ack into
    *charlie-work's* state file — exit 0, no warning, and the sibling repo's finding
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


#: Commands that mutate orchestrator state and are rare, operator-driven, and
#: high-cost-of-misroute.  ``_assert_not_sibling_clone`` is called for these
#: only (issue #1376): a wrong-cwd invocation silently writes a valid record
#: into a sibling clone's phantom ``.var`` tree, invisible to the canonical
#: repo — and if the target is a transient dispatch/agent worktree, the reap
#: lanes later delete the whole checkout, destroying the record permanently.
#: Read-only commands are deliberately exempt: their cwd-defaulted resolution
#: is harmless and changing it would break operator workflows that routinely
#: run ``charlie status`` from worktree cwds.
_STATE_AFFECTING_COMMANDS = frozenset({"verdict", "merge-authorize", "unescalate"})


def _assert_not_sibling_clone(ctx: CommandContext, args: argparse.Namespace) -> None:
    """Refuse to write state from a sibling clone of the canonical fleet repo.

    The DEFAULT-case sibling of the documented ``--config`` trap
    (``_assert_config_repo_matches``, issue #895): there, an explicit flag
    misleads; here, no flag at all plus an unexpected cwd silently selects a
    different state root.  ``find_repo_root`` already resolves linked
    worktrees to the shared main checkout (issue #648), so a verdict run from
    a ``.claude/worktrees/*`` or agent worktree correctly targets the
    canonical ``.var/charlie-work``.  The remaining hazard is a **sibling
    clone** — a separate git repo (e.g. ``repos/cw-*``) that shares the same
    GitHub remote but has its own ``.git`` and therefore its own phantom
    ``.var/charlie-work``.  ``find_repo_root`` returns the clone's own root
    (it is the main worktree of its own repo), and state silently lands there.

    Detection uses the fleet registry: the canonical repo is registered under
    its ``nameWithOwner`` with a ``repo_root`` pointing at the main checkout.
    If the current ``repo_root`` differs from the registered one (after
    normalizing both through ``find_repo_root`` so an old entry pointing at a
    linked worktree resolves to the same shared root), the current cwd is a
    sibling clone and the command is refused.

    Fails open (allows the command) when:
    - ``nameWithOwner`` cannot be resolved from ``git remote get-url origin``
      (non-GitHub repo, no origin remote) — the guard is for GitHub-fleet
      repos, not arbitrary git checkouts.
    - The repo is not yet in the fleet registry (fresh install) — there is no
      canonical root to compare against.
    - The registered ``repo_root`` no longer exists or is not a git worktree
      — a stale entry is not evidence of a sibling clone.

    Explicit ``--repo`` skips this guard entirely (acceptance criterion #4):
    the operator named the repo, so the resolution is intentional.
    """
    # Resolve the repo's GitHub identity from the local git remote — no
    # network round-trip, works under --dry-run and in tests with real repos.
    try:
        owner, name = ctx.gh._repo_owner_name()
    except GitHubError:
        return
    name_with_owner = f"{owner}/{name}"

    fleet_json_path = layout.fleet_registry_path(override=args.fleet_dir)
    registry = _load_registry(fleet_json_path)
    entry = registry.get("repos", {}).get(name_with_owner)
    if not entry:
        return

    registered_root_str = entry.get("repo_root")
    if not registered_root_str:
        return
    registered_root = Path(registered_root_str)
    if not registered_root.exists():
        return

    # Normalize the registered root through find_repo_root so an old entry
    # pointing at a linked worktree resolves to the shared main checkout —
    # the same normalization find_repo_root applies to cwd.  Without this,
    # a pre-#692 registry entry would false-positive as a sibling clone.
    try:
        normalized_registered = find_repo_root(registered_root, explicit=True)
    except RepoNotFoundError:
        return

    if ctx.repo_root == normalized_registered:
        return

    cwd = Path.cwd()
    raise ConfigError(
        f"cwd {cwd} resolves to repo root {ctx.repo_root}, whose state root "
        f"is {ctx.paths.root}. The fleet registry has {name_with_owner} "
        f"registered at {normalized_registered} — the canonical root. "
        f"State would silently land in the sibling clone's tree, invisible "
        f"to the canonical repo. "
        f"Pass --repo {normalized_registered} to operate on the canonical repo, "
        f"or cd to {normalized_registered}."
    )


@dataclass(frozen=True)
class CommandContext:
    """The four bootstrap artifacts every command handler needs (issue #705).

    Centralizes the ``find_repo_root`` -> ``load_layered_config`` ->
    ``runtime_paths`` -> ``GitHub(...)`` sequence so a new command handler
    cannot get the bootstrap order or arguments wrong — there is only one
    way to call it. Frozen to match the project invariant that config/value
    objects are immutable.
    """

    repo_root: Path
    config: OrchestratorConfig
    paths: RuntimePaths
    gh: GitHub


def bootstrap_command(
    args: argparse.Namespace,
    *,
    redirect_to_main_worktree: bool = True,
) -> CommandContext:
    """Run the four-call CLI bootstrap once, returning a frozen context.

    This is the single shared entry point for the
    ``find_repo_root`` -> ``_assert_config_repo_matches`` ->
    ``load_layered_config`` -> ``runtime_paths`` -> ``GitHub(...)`` sequence
    that every command handler needs (issue #705).  Previously each handler
    reproduced the calls independently, with no compiler or test enforcing
    the order — a new handler that forgot one call silently bootstrapped
    against the wrong repo root / config layer / runtime paths.

    ``_assert_config_repo_matches`` (issue #895) is included here so every
    command inherits the config-vs-state misroute guard, not only
    ``build_app``.  ``run_runners_allocate`` is the one exception: it uses
    ``require_global=True`` with custom error handling and cannot use this
    helper.

    When *redirect_to_main_worktree* is False the linked-worktree redirect in
    :func:`find_repo_root` is skipped, so a read-only diagnostic invoked from
    a linked worktree bootstraps against that worktree's own root (issue
    #1600).  State-mutating commands must keep the default ``True`` so their
    state resolves to the shared ``.var/charlie-work/`` directory (issue #648).
    """
    repo_root = find_repo_root(
        args.repo,
        explicit=args.repo is not None,
        redirect_to_main_worktree=redirect_to_main_worktree,
    )
    _assert_config_repo_matches(args.config, repo_root)
    config = load_layered_config(repo_root, args.config, fleet_dir_override=args.fleet_dir)
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    gh = GitHub(repo_root=repo_root, runtime=config.runtime, dry_run=args.dry_run)
    return CommandContext(repo_root=repo_root, config=config, paths=paths, gh=gh)


def build_app(args: argparse.Namespace) -> OrchestratorApp:
    ctx = bootstrap_command(args)
    # Issue #1376: for state-affecting commands, refuse before touch_repo
    # mutates the fleet registry — a sibling-clone cwd would otherwise
    # overwrite the canonical entry and the guard would never fire.  Explicit
    # --repo skips the guard (the operator named the repo intentionally).
    # `args.repo is None` is checked first, and `command` is read defensively
    # via getattr: build_app is also called directly (outside the full
    # argparse pipeline) by tests and other callers that hand-build a
    # Namespace without a `command` attribute. Ordering the cheap, always-
    # present `repo` check first avoids an AttributeError on `args.command`
    # for those callers when --repo is explicit (AC#4 already skips the
    # guard in that case), and getattr keeps the check inert rather than
    # crashing if `command` is absent entirely.
    if args.repo is None and getattr(args, "command", None) in _STATE_AFFECTING_COMMANDS:
        _assert_not_sibling_clone(ctx, args)
    touch_repo(args.fleet_dir, ctx.repo_root, ctx.paths, ctx.gh, dry_run=args.dry_run)
    return OrchestratorApp(
        ctx.repo_root,
        ctx.paths,
        ctx.config,
        ctx.gh,
        dry_run=args.dry_run,
        fleet_dir_override=args.fleet_dir,
    )


def run_doctor_command(args: argparse.Namespace) -> CommandResult:
    # #6-G: doctor exists to diagnose a broken repo, so it must not itself
    # crash on the exact condition it is meant to diagnose (e.g. the
    # unknown-config-key ConfigError that silently killed the cw fleet lane
    # three times on 2026-07-29). Render the parse failure as a structured
    # finding instead of letting it propagate to main()'s generic
    # `except (ConfigError, ValueError)` handler, which prints to stderr and
    # exits 2 with no machine-readable finding at all.
    try:
        ctx = bootstrap_command(args)
    except ConfigError as exc:
        # bootstrap_command calls find_repo_root before the failing call, so
        # re-running it here for the error-message path is safe and
        # deterministic.  Doctor is not performance-sensitive.
        repo_root = find_repo_root(args.repo, explicit=args.repo is not None)
        config_path = find_config_path(repo_root, args.config)
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
    config_path = find_config_path(ctx.repo_root, args.config)
    touch_repo(args.fleet_dir, ctx.repo_root, ctx.paths, ctx.gh, dry_run=args.dry_run)
    ok, checks = run_doctor(
        ctx.repo_root,
        ctx.paths,
        ctx.config,
        config_path,
        ctx.gh,
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
    ctx = bootstrap_command(args)
    state = load_state_locked(ctx.paths.state_file)
    result = clean_worktrees(
        ctx.repo_root,
        resolved_layout(ctx.config, ctx.repo_root).worktrees,
        state,
        ctx.config,
        ctx.gh,
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
    ctx = bootstrap_command(args)

    pr = ctx.gh.pr_view(args.pr, fields=CLOSING_KEYWORD_PR_FIELDS)
    if not pr:
        return CommandResult(False, f"closing-keyword-check: could not fetch PR #{args.pr}", {})

    commits = ctx.gh.pr_commits(args.pr)
    if commits is None:
        return CommandResult(
            False, f"closing-keyword-check: could not fetch commits for PR #{args.pr}", {}
        )
    commit_messages = [str((c.get("commit") or {}).get("message") or "") for c in commits]

    # Issue #1229 scoping decision: this call site is deliberately NOT
    # threaded through branch_issue_validator. ``intended`` is the single
    # issue number ``find_unexpected_closing_references`` exempts from its
    # unexpected-closing-reference scan; it is a diagnostic/reporting value
    # (surfaced as ``intended_issue_number`` in the command's JSON output),
    # not a key for any issue-label transition or state write. A stale
    # branch-name binding would set ``intended`` to the wrong number, causing
    # the real intended issue's closing keyword to be flagged as an
    # unexpected reference -- a conservative false-positive failure direction
    # (the check blocks rather than corrupts), and one an operator can
    # resolve by rewording the PR body. Threading the validator would also
    # add an ``issue_list(state="open")`` call to a one-shot CLI command that
    # otherwise makes only the two ``pr_view``/``pr_commits`` calls above.
    intended = linked_issue_number(
        pr,
        is_cross_repository=pr.get("isCrossRepository"),
        branch_prefix=ctx.config.dispatch.branch_prefix,
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


def run_mojibake_check_command(args: argparse.Namespace) -> CommandResult:
    """CI gate (issue #1057): fail if the diff introduces mojibake.

    Runs ``git diff <base>..HEAD`` in the repo root and scans every added
    line for cp1252/UTF-8 mojibake via :func:`find_mojibake_in_diff`.  The
    detection is derived from the encoding process (reverse the corruption
    and check whether the result differs) rather than a hardcoded list of
    bad byte sequences, so it catches any UTF-8/cp1252 round trip -- not
    just the specific em-dash sequence documented in the issue.

    Uses a two-dot diff (``base..HEAD``) rather than three-dot
    (``base...HEAD``) because CI runs against a shallow clone
    (``actions/checkout@v5`` with ``fetch-depth: 1``).  Three-dot needs the
    merge-base of *base* and HEAD, which requires traversing the ancestry
    chain between them -- impossible when the shallow boundary cuts it.
    Two-dot compares the two trees directly (no merge-base computation) and
    works once both commits are present.  The CI workflow fetches the base
    SHA with ``git fetch --depth 1`` before invoking this command; see the
    "Mojibake gate" step in ``ci.yml``.

    Like the closing-keyword gate, this is deliberately a step of the
    existing "Lint" job (added in ci.yml), not a new job: GitHub reports
    check-run status per job, so riding the already-required "Lint" context
    makes this a de facto blocking gate the moment a PR branch includes the
    workflow change -- no branch-protection edit, no orchestrator.config.yaml
    change, no separate promotion step.

    Errors as values (per CLAUDE.md): a git failure comes back as
    ``CommandResult(ok=False)`` -- never raised -- so the CI step exits
    non-zero without a Python traceback.
    """
    ctx = bootstrap_command(args)

    base = getattr(args, "base", "origin/main")
    result = run_captured(
        ["git", "diff", f"{base}..HEAD"],
        cwd=ctx.repo_root,
        timeout_seconds=60,
    )
    if not result.ok:
        return CommandResult(
            False,
            f"mojibake-check: could not run git diff against {base}: "
            f"{result.error or result.stderr or 'git diff failed'}",
            {"base": base},
        )

    findings = find_mojibake_in_diff(result.stdout)

    data = {
        "base": base,
        "findings": [
            {
                "path": f.path,
                "line": f.line_number,
                "content": f.content,
                "recovered": f.recovered,
            }
            for f in findings
        ],
    }

    if findings:
        lines = [f"  {f.path}:{f.line_number}: {f.content!r} -> {f.recovered!r}" for f in findings]
        message = (
            f"mojibake-check: {len(findings)} corrupted line(s) in diff "
            f"against {base}\n"
            + "\n".join(lines)
            + "\nNon-ASCII characters were corrupted by a UTF-8/cp1252 round "
            "trip. Restore the original characters -- do NOT replace them with "
            "ASCII equivalents."
        )
        return CommandResult(False, message, data)

    return CommandResult(
        True,
        f"mojibake-check: clean (diff against {base})",
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
    dirty_tree_checker=check_working_tree_clean,
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

    Clean-tree gate (issue #729): ``--apply`` also refuses when the tracked
    working tree differs from ``HEAD``, because the command executes the
    working tree while CI only reviewed the committed tree -- a guard neutered
    only in the working tree is invisible to every review and test run that
    validated the commit. ``--allow-dirty`` overrides this for deliberate
    local testing. Plan-only and dry-run paths never reach this gate, since
    iterating on a plan against a dirty tree is the normal development loop.
    """
    ctx = bootstrap_command(args)

    src_root = (
        _resolve_migration_root(args.src, ctx.repo_root)
        if args.src is not None
        else ctx.paths.root
    )
    dst_root = (
        _resolve_migration_root(args.dst, ctx.repo_root)
        if args.dst is not None
        # ``.resolve()`` here (not just in ``layout.default_state_root``) matters:
        # ``runtime_paths`` above resolves symlinks/junctions in the *whole* src
        # path, not just ``repo_root``. Without a matching resolve on this side,
        # a repo whose ``.var`` is itself a symlink/junction would make the two
        # roots compare unequal even when they name the same on-disk location,
        # producing a same-place migration plan instead of the intended
        # already-migrated short-circuit. Safe on a not-yet-created dst: Path
        # .resolve() does not require the path to exist.
        else layout.default_state_root(ctx.repo_root).resolve()
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

    plan = planner(repo_root=ctx.repo_root, src_root=src_root, dst_root=dst_root)
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

    # Issue #729: the command executes the working tree, but CI only reviewed
    # the committed tree. Refuse to actuate when the tracked working tree
    # differs from HEAD, naming the divergent paths so the operator sees *what*
    # changed rather than just being blocked. ``--allow-dirty`` overrides this
    # for deliberate local testing. A probe that cannot determine cleanliness
    # (git failed) is also refused -- fail-closed, never silently proceed.
    if not args.allow_dirty:
        dirty = dirty_tree_checker(repo_root=ctx.repo_root)
        if not dirty.ok:
            return CommandResult(
                False,
                f"{rendered}\nrefusing to apply: {dirty.error}",
                data,
            )
        if not dirty.clean:
            paths = "\n".join(f"  {p}" for p in dirty.dirty_paths)
            return CommandResult(
                False,
                f"{rendered}\nrefusing to apply: tracked working tree differs from "
                f"HEAD ({len(dirty.dirty_paths)} path(s)); the command executes the "
                f"working tree but CI only reviewed the committed tree:\n{paths}\n"
                "pass --allow-dirty to override for deliberate local testing",
                data,
            )

    outcome = actuator(plan)
    data = {
        **data,
        "applied": outcome.ok,
        "moved": list(outcome.moved),
        "rewritten_paths": outcome.rewritten_paths,
    }
    if not outcome.ok:
        data = {**data, "aborted_at": outcome.aborted_at}
        return CommandResult(
            False,
            f"{rendered}\nmigration failed after {len(outcome.moved)} moved: {outcome.error}",
            data,
        )
    return CommandResult(
        True,
        f"{rendered}\nmoved {len(outcome.moved)} children, "
        f"rewrote {outcome.rewritten_paths} embedded paths",
        data,
    )


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
    state_dir = (
        global_config.runtime.state_dir if global_config is not None else layout.DEFAULT_STATE_DIR
    )
    state_root = runtime_paths(orchestrator_root(), state_dir).root
    deploy = self_deploy(
        orchestrator_root(),
        state_root=state_root,
        fleet_dir_override=args.fleet_dir,
        dry_run=args.dry_run,
        pull_ci_fleet=(
            global_config.supervisor.self_deploy_pull_ci_fleet
            if global_config is not None
            else False
        ),
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
    # Issue #1372: stale entries (repo_root no longer exists) are reported in a
    # separate "stale" list that does NOT flip ok/exit-code, so one corpse
    # cannot degrade fleet-wide tooling (e.g. the heartbeat's blocked-issue
    # enrichment that treats any nonzero exit as degraded).
    stale: list[dict[str, str]] = []

    for repo_key, entry in sorted(registry.get("repos", {}).items()):
        try:
            repo_root = Path(entry.get("repo_root") or "")
            if not repo_root.exists():
                # Issue #1372: a stale entry is not a live failing lane —
                # report it separately so it does not affect the exit code.
                stale.append({"repo_key": repo_key, "repo_root": str(repo_root)})
                continue

            config = load_layered_config(repo_root, None, fleet_dir_override=args.fleet_dir)
            paths = runtime_paths(repo_root, config.runtime.state_dir)
            gh = GitHub(repo_root=repo_root, runtime=config.runtime, dry_run=True)
            app = OrchestratorApp(repo_root, paths, config, gh, dry_run=True)
            result = app.status(use_cache=not getattr(args, "no_cache", False))
            per_repo[repo_key] = result.data
        except (RepoNotFoundError, ConfigError, GitHubError, OSError) as exc:
            errors.append({"repo_key": repo_key, "error": str(exc)})

    # api-worker fleet report line (issue #483): read-only, never raises.
    api_worker_report = compute_api_worker_fleet_report(fleet_dir_override=args.fleet_dir)

    return CommandResult(
        ok=not errors,
        message=f"fleet status: {len(per_repo)} repo(s), {len(errors)} error(s), {len(stale)} stale(s)",
        data={
            "repos": per_repo,
            "errors": errors,
            "stale": stale,
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


def run_fleet_operator_queue(args: argparse.Namespace) -> CommandResult:
    """Run fleet operator-queue aggregation across all registered repos.

    Issue #1314 item 1. This is a read-only command that mirrors
    ``run_fleet_review_queue``: for each registered repo, calls
    ``OrchestratorApp.operator_queue()`` with ``dry_run=True``, aggregates
    per-repo queue entries keyed by repo_key (nameWithOwner), and isolates
    per-repo errors without aborting aggregation.
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
            result = app.operator_queue()
            per_repo[repo_key] = result.data
        except (RepoNotFoundError, ConfigError, GitHubError, OSError) as exc:
            errors.append({"repo_key": repo_key, "error": str(exc)})

    return CommandResult(
        ok=not errors,
        message=f"fleet operator queue: {len(per_repo)} repo(s), {len(errors)} error(s)",
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
    ctx = bootstrap_command(args)

    if not ctx.config.runner_scaling.enabled:
        return CommandResult(
            ok=False,
            message="runner_scaling feature is not enabled in config",
            data={},
        )

    try:
        pool_state = observe_runner_pool(
            ctx.gh, ctx.config.runner_scaling, state_dir=ctx.paths.root, dry_run=args.dry_run
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
    from ci_fleet.runner_allocation import SlotAction
    from ci_fleet.shadow_gate import (
        REQUIRED_CALENDAR_DAYS,
        REQUIRED_STREAK,
        evaluate as evaluate_shadow_gate,
    )

    ctx = bootstrap_command(args)

    data: dict[str, Any] = {}

    # --- Store 1: events.db (this repo's state dir) -----------------------
    # Same location instrumentation.py's _db_path derives: state.json's
    # parent, i.e. paths.root. Checked for existence before doing anything
    # else so a missing DB is reported as missing rather than silently
    # created -- sqlite3.connect() would otherwise create an empty file the
    # instant query_events() opened it, which is not a "read-only reporter"
    # for a store that was never written.
    events_db_path = ctx.paths.root / "events.db"
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
        allocation_events = query_events(ctx.paths.state_file, kind="runner_allocation")
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
        data["change_agreement_streak_by_action"] = None
        data["provisioning_action"] = None
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

    # Per-action split of the load-bearing streak (#926). Counts individual
    # changes, not passes, because one pass can contain several decisions. The
    # action for a change is taken only where both planners agree on the exact
    # (repo, runner, action) tuple -- a disagreement about the kind of change
    # is the finding and must not be collapsed into a bucket.
    def _change_action_tuples(rec: Mapping[str, Any]) -> set[tuple[str, str, str]]:
        live = (rec.get("live_plan") or {}).get("changes") or []
        shadow = (rec.get("shadow_plan") or {}).get("changes") or []
        live_set = {(c["repo"], c["runner"], c["action"]) for c in live}
        shadow_set = {(c["repo"], c["runner"], c["action"]) for c in shadow}
        return live_set & shadow_set

    def _action_counts(recs: Iterable[Mapping[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rec in recs:
            for _repo, _runner, action in _change_action_tuples(rec):
                counts[action] = counts.get(action, 0) + 1
        return counts

    # The trailing run of changed passes that all agreed (same suffix that
    # underlies change_agreement_streak, but we need the records, not just a
    # count, to tally the actions inside them).
    trailing_changed_records: list[Mapping[str, Any]] = []
    for rec in reversed(changed_records):
        if not rec.get("agreed"):
            break
        trailing_changed_records.append(rec)

    total_action_counts = _action_counts(changed_records)
    streak_action_counts = _action_counts(reversed(trailing_changed_records))

    # Always show both SlotAction values so a 0/0 action is surfaced even when
    # it has never been observed in the journal.
    known_actions = {SlotAction.PARK.value, SlotAction.START.value}
    observed_actions = set(total_action_counts) | set(streak_action_counts)
    all_actions = known_actions | observed_actions

    data["change_agreement_streak_by_action"] = {
        action: {
            "streak": streak_action_counts.get(action, 0),
            "total": total_action_counts.get(action, 0),
        }
        for action in sorted(all_actions)
    }
    data["provisioning_action"] = SlotAction.START.value

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
    ctx = bootstrap_command(args)

    if not ctx.config.runner_scaling.enabled:
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
    if ctx.config.runner_allocation.enabled and not force:
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

    if not ctx.config.runner_scaling.managed_root:
        return CommandResult(
            ok=False,
            message="runner_scaling.managed_root is not configured",
            data={},
        )

    managed_root = Path(ctx.config.runner_scaling.managed_root)
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
        ctx.config.runner_scaling.runner_dir_prefix,
        ctx.config.runner_scaling,
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
    ctx = bootstrap_command(args)

    if not ctx.config.runner_scaling.enabled:
        return CommandResult(
            ok=False,
            message="runner_scaling feature is not enabled in config",
            data={},
        )

    if not ctx.config.runner_scaling.managed_root:
        return CommandResult(
            ok=False,
            message="runner_scaling.managed_root is not configured",
            data={},
        )

    managed_root = Path(ctx.config.runner_scaling.managed_root)
    if not managed_root.exists():
        return CommandResult(
            ok=False,
            message=f"managed_root does not exist: {managed_root}",
            data={},
        )

    # Use subparser-specific dry_run flag if available, otherwise fall back to global
    dry_run = getattr(args, "dry_run", False)

    removed_count, errors = scale_down_idle_runners(
        managed_root,
        ctx.config.runner_scaling.runner_dir_prefix,
        ctx.gh,
        ctx.config.runner_scaling,
        ctx.paths.root,
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
    ctx = bootstrap_command(args)

    if not ctx.config.runner_scaling.enabled:
        return CommandResult(
            ok=False,
            message="runner_scaling feature is not enabled in config",
            data={},
        )

    # Use subparser-specific dry_run flag if available, otherwise fall back to global
    dry_run = getattr(args, "dry_run", False)
    fleet_wide = getattr(args, "fleet_wide", False)

    # Observe current pool state
    state = observe_runner_pool(
        ctx.gh, ctx.config.runner_scaling, state_dir=ctx.paths.root, dry_run=dry_run
    )

    # Load fleet-wide totals if requested
    fleet_totals: FleetTotals | None = None
    skipped_repos: list[str] = []
    if fleet_wide:
        total_runners, total_busy_runners, skipped_repos = count_fleet_runners(
            args.fleet_dir, runtime=ctx.config.runtime
        )
        fleet_totals = FleetTotals(
            total_runners=total_runners,
            total_busy_runners=total_busy_runners,
        )

    # Check cooldown and idle duration
    in_cooldown = is_in_cooldown(ctx.paths.root, ctx.config.runner_scaling.cooldown_minutes)
    is_idle_for_duration = is_pool_idle_for_minutes(
        ctx.paths.root, ctx.config.runner_scaling.idle_scale_down_minutes
    )

    # Run the pure decision function
    decision = decide_autoscale(
        state,
        ctx.config.runner_scaling,
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

        # Affinity knobs (issue: ci_runners #92 companion) — sourced from the
        # same runner_allocation section launch_runner_listener's callers use,
        # never hardcoded. 0/0 (the section's defaults) is a no-op downstream.
        # Requires ci_runners #92 merged and deployed: the currently installed
        # ci_fleet.provision_runner does not yet accept these kwargs.
        result = provision_runner(
            ctx.gh,
            ctx.config.runner_scaling,
            state.busy_runners,
            dry_run=False,
            reserved_threads=ctx.config.runner_allocation.reserved_threads,
            threads_per_slot=ctx.config.runner_allocation.threads_per_slot,
        )
        if result.ok:
            # Record scale event
            from ci_fleet.charlie_work_adapter import record_scale_event

            record_scale_event(ctx.paths.root, "up")
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
        managed_root = Path(ctx.config.runner_scaling.managed_root)
        if not managed_root.exists():
            return CommandResult(
                ok=False,
                message=f"managed_root does not exist: {managed_root}",
                data={},
            )

        removed_count, errors = scale_down_idle_runners(
            managed_root,
            ctx.config.runner_scaling.runner_dir_prefix,
            ctx.gh,
            ctx.config.runner_scaling,
            ctx.paths.root,
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


def run_runners_provision(args: argparse.Namespace) -> CommandResult:
    """Manually provision one new runner when the pool is starved (issue #826).

    Scale-up only. Reuses ``decide_autoscale()``'s guardrails — max_runners,
    RAM headroom, cooldown, CPU threshold — so the manual trigger cannot
    bypass a ceiling that the unattended path would respect. Deliberately
    never scales down: ``provision`` is an "add capacity" button, not a
    second autoscale. If ``decide_autoscale`` returns ``ScaleAction.DOWN``,
    the command reports the decision and exits without action.

    Operator ruling (2026-08-13): approved as manual-trigger only, NOT
    unattended autoscale. ``runner_scaling.enabled`` remaining false (the
    code default) is a hard refusal — the operator must opt in by setting
    the ``runner_scaling`` section (``enabled``, ``managed_root``,
    ``package_zip``) in config before this command can act.

    Provisioning stays on its own cadence and must not start or stop
    listeners — ``charlie runners allocate`` remains the only controller of
    which listeners run. This command only adds registrations via
    ``provision_runner``; it never calls ``scale_down_idle_runners``.
    """
    ctx = bootstrap_command(args)

    if not ctx.config.runner_scaling.enabled:
        return CommandResult(
            ok=False,
            message="runner_scaling feature is not enabled in config",
            data={},
        )

    dry_run = getattr(args, "dry_run", False)
    fleet_wide = getattr(args, "fleet_wide", False)

    # Observe current pool state
    state = observe_runner_pool(
        ctx.gh, ctx.config.runner_scaling, state_dir=ctx.paths.root, dry_run=dry_run
    )

    # Load fleet-wide totals if requested — same guardrail source as autoscale
    fleet_totals: FleetTotals | None = None
    skipped_repos: list[str] = []
    if fleet_wide:
        total_runners, total_busy_runners, skipped_repos = count_fleet_runners(
            args.fleet_dir, runtime=ctx.config.runtime
        )
        fleet_totals = FleetTotals(
            total_runners=total_runners,
            total_busy_runners=total_busy_runners,
        )

    # Check cooldown — a guardrail decide_autoscale also enforces internally,
    # but surfacing it here lets the operator see the reason without digging
    # through the decision's ``reason`` string.
    in_cooldown = is_in_cooldown(ctx.paths.root, ctx.config.runner_scaling.cooldown_minutes)

    # is_idle_for_duration is a scale-down input only. Provision is scale-up
    # only, so it is always False here — passing True would let decide_autoscale
    # return DOWN, which this command would then refuse to act on anyway.
    # Hardcoding False is not a hardcoded element (rule #9): it is the correct
    # value for an input this command does not use, not a list of things to
    # manage.
    decision = decide_autoscale(
        state,
        ctx.config.runner_scaling,
        fleet_totals=fleet_totals,
        in_cooldown=in_cooldown,
        is_idle_for_duration=False,
    )

    # Provision is scale-up only. A DOWN decision is reported but not acted
    # on — this command must not become a second scale-down path that fights
    # allocation's hysteresis or autoscale's idle detection.
    if decision.action == ScaleAction.DOWN:
        return CommandResult(
            ok=True,
            message=(
                f"provision: scale-down declined (provision is scale-up only) - {decision.reason}"
            ),
            data={
                "decision": {
                    "action": decision.action.value,
                    "count": decision.count,
                    "reason": decision.reason,
                },
                "declined": True,
            },
        )

    # In dry-run mode, return the decision without executing
    if dry_run or decision.action != ScaleAction.UP:
        return CommandResult(
            ok=True,
            message=f"provision: no action - {decision.reason}",
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
                    "skipped_repos": skipped_repos,
                }
                if fleet_totals
                else None,
            },
        )

    # Execute the scale-up
    from ci_fleet.charlie_work_adapter import provision_runner

    # Affinity knobs sourced from runner_allocation, same as autoscale.
    # 0/0 (the section's defaults) is a no-op downstream.
    result = provision_runner(
        ctx.gh,
        ctx.config.runner_scaling,
        state.busy_runners,
        dry_run=False,
        reserved_threads=ctx.config.runner_allocation.reserved_threads,
        threads_per_slot=ctx.config.runner_allocation.threads_per_slot,
    )
    if result.ok:
        from ci_fleet.charlie_work_adapter import record_scale_event

        record_scale_event(ctx.paths.root, "up")
        return CommandResult(
            ok=True,
            message=f"provision: scaled up - {decision.reason}",
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
            message=f"provision: scale up failed - {result.error}",
            data={
                "decision": {
                    "action": decision.action.value,
                    "count": decision.count,
                    "reason": decision.reason,
                },
                "error": result.error,
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

    # Issue #927: `charlie runners allocate` is the operator path to the same
    # actuation the supervisor prologue guards -- run_allocation_pass_with_ci_fleet_guard
    # is the single point of enforcement, so this command inherits the dirty-worktree
    # guard by construction instead of needing its own copy of the check.
    result, dirty_check = run_allocation_pass_with_ci_fleet_guard(
        gh,
        config.runner_allocation,
        managed_root_fallback=config.runner_scaling.managed_root,
        fleet_dir_override=args.fleet_dir,
        state_path=paths.state_file,
        dry_run=dry_run,
        source=CLI_ALLOCATION_SOURCE,
        full_pass_interval_seconds=config.supervisor.full_pass_interval_seconds,
    )
    dry_run_forced = dirty_check.is_dirty
    effective_dry_run = dry_run or dry_run_forced

    if result.error:
        return CommandResult(ok=False, message=f"allocate: {result.error}", data={})

    data: dict[str, Any] = {
        "dry_run": effective_dry_run,
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
    if dry_run_forced:
        data["ci_fleet_worktree_dirty"] = {
            "ci_fleet_root": str(dirty_check.repo_root),
            "dirty_paths": list(dirty_check.dirty_paths),
        }
    if result.plan is not None:
        data["plan"] = plan_summary(result.plan)

    if result.skipped:
        return CommandResult(
            ok=True,
            message=f"allocate: no action - {'; '.join(result.notes) or 'nothing to do'}",
            data=data,
        )

    if dry_run_forced:
        prefix = "would "
        dirty_paths_text = "; ".join(dirty_check.dirty_paths)
        guard_suffix = (
            f" (forced dry-run: ci_fleet dependency tree at {dirty_check.repo_root} "
            f"has uncommitted changes: {dirty_paths_text})"
        )
    else:
        prefix = "would " if dry_run else ""
        guard_suffix = ""
    return CommandResult(
        ok=result.ok,
        message=(
            f"allocate: {prefix}start {result.started}, {prefix}park {result.parked} "
            f"(budget {result.plan.budget if result.plan else 0})" + guard_suffix
        ),
        data=data,
    )


def run_command(app: OrchestratorApp, args: argparse.Namespace) -> CommandResult:
    if args.command == "roll-call":
        return app.status(use_cache=not getattr(args, "no_cache", False))
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
    if args.command == "operator-queue":
        return app.operator_queue()
    if args.command == "why-charlie-hate":
        return app.review(args.pr)
    if args.command == "verdict":
        try:
            return app.record_review(
                args.pr,
                args.decision,
                summary=args.summary,
                summary_file=args.summary_file,
                comment=args.comment,
                reviewed_head=args.reviewed_head,
                # Issue #1265: a human running this command is, by
                # definition, the operator-manual provenance -- no flag to
                # thread through, this is the one caller for which the value
                # is always the same.
                verdict_provenance="operator_manual",
                # Issue #1072: the operator CLI is the one caller that may
                # legitimately pin a verdict to a superseded head (issue #467's
                # explicit-choice design). Automated callers use the default
                # False and are refused by record_review()'s compare-and-swap
                # guard when the live head has moved past the packet head.
                allow_stale_head=True,
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
    if args.command == "merge-authorize":
        return app.merge_authorize(args.pr, args.reason or "", by=args.by, sha=args.sha)
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


def _render_backlog_reachability(reachability: Any) -> str:
    """Issue #944: one-line suffix saying why a repo has nothing to dispatch.

    The counts on the fleet-status line all derive from the ready-FILTERED
    issue query, so "nothing to do" and "the whole backlog is unreachable"
    render identically as zeroes. Between 2026-07-31 and 2026-08-05 that
    ambiguity hid a total dispatch stall: 87 open issues, 0 dispatchable, four
    days, every pass green.

    Returns "" when there is nothing to say -- a healthy repo with work
    flowing keeps its line unchanged. ASCII only: the console codepage here is
    cp437.
    """
    if not isinstance(reachability, dict):
        return ""
    if not reachability.get("observed"):
        # NOT the same as "0 open issues": the unfiltered fetch returned
        # nothing, and an empty list is indistinguishable from a failed `gh`
        # call. Saying "0 open" here would be the very error this exists to
        # catch. See workflow.classify_backlog_reachability.
        return "  [backlog not observed]"

    notes: list[str] = []
    # `is False` deliberately, not falsiness: `consistent` is tri-state and
    # None means the cross-check never ran. Neither "passed" nor "failed" is
    # an honest rendering of that, and it cannot reach here anyway -- the
    # unobserved path returns above.
    if reachability.get("consistent") is False:
        notes.append("INCONSISTENT backlog fetch (missing ready-labelled issues)")

    open_total = int(reachability.get("open_total") or 0)
    dispatchable = int(reachability.get("dispatchable") or 0)
    if open_total and not dispatchable:
        reasons = ", ".join(
            f"{reason}={reachability[reason]}"
            for reason in (
                "missing_ready",
                "terminal_label",
                "active_label",
                "operator_claimed",
                "blocked_by_open_dependency",
                "mention_covered_awaiting_operator",
                "unidentified",
            )
            if reachability.get(reason)
        )
        # The bins partition the backlog, so `reasons` is non-empty whenever
        # open_total is. Kept as a guard rather than an assumption: an alarm
        # that fires without naming a cause is worse than no alarm.
        if not reasons:
            reasons = "no reason recorded -- classifier bug"
        notes.append(f"!! {open_total} open, 0 dispatchable ({reasons})")

    return "  " + "; ".join(notes) if notes else ""


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
            elif args.fleet_command == "operator-queue":
                result = run_fleet_operator_queue(args)
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
            elif args.runners_command == "provision":
                result = run_runners_provision(args)
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
        elif args.command == "mojibake-check":
            result = run_mojibake_check_command(args)
        elif args.command == "private-slug-check":
            result = run_private_slug_check_command(args)
        elif args.command == "ast-equivalence-check":
            result = run_ast_equivalence_check_command(args)
        elif args.command == "junit-recorded-check":
            result = run_junit_recorded_check_command(args)
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
                line = (
                    f"  {repo_key}: {ready} ready, {active} active, "
                    f"{blocked} blocked, {stalled} stalled"
                )
                # Issue #944: every count above is derived from the
                # ready-FILTERED issue query, so a repo with 87 open issues and
                # none reachable prints identically to a repo with nothing to
                # do. That ambiguity hid a four-day total dispatch stall. Say
                # which one it is.
                suffix = _render_backlog_reachability(repo_data.get("backlog_reachability"))
                print(line + suffix)
                cache_age = repo_data.get("cache_age_seconds")
                if cache_age is not None:
                    print(f"    (cached, {cache_age:.0f}s old)")
            errors = result.data.get("errors", [])
            if errors:
                print("Errors:")
                for error in errors:
                    print(f"  {error['repo_key']}: {error['error']}")
            # Issue #1372: stale entries are reported separately and do not
            # affect the exit code; surface them so an operator can see and
            # clean up corpses without mistaking them for live failing lanes.
            stale = result.data.get("stale", [])
            if stale:
                print("Stale:")
                for entry in stale:
                    print(f"  {entry['repo_key']}: {entry['repo_root']}")
            _render_api_worker_report(result.data)
        elif args.fleet_command in ("work", "bash-rats"):
            repos = result.data.get("repos", {})
            for repo_key, repo_data in sorted(repos.items()):
                # repo_data now includes the ok field from fleet_dispatch
                ok = repo_data.get("ok", True)
                status = "OK" if ok else "FAILED"
                if repo_data.get("pass_skipped"):
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
                by_action = result.data.get("change_agreement_streak_by_action") or {}
                provisioning_action = result.data.get("provisioning_action") or "start"
                for action, info in sorted(by_action.items()):
                    streak = info.get("streak", 0)
                    total = info.get("total", 0)
                    if total == 0:
                        if action == provisioning_action:
                            note = " <- provisioning path, never compared"
                        else:
                            note = " <- never compared"
                    else:
                        note = ""
                    print(f"      {action}: {streak}/{total}{note}")
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
                        f"compared {change_streak.get('total')} time(s). "
                        "Per-action counts are individual changes, not passes."
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

    # Issue #1363: a fatal preflight failure at supervisor startup (disk
    # floor, wrong venv/checkout) exits PREFLIGHT_REFUSAL_EXIT_CODE (4), not
    # the generic 1 -- so the fleet pass log and the supervise-loop wrapper
    # can both distinguish "refused to start, named reason" from an ordinary
    # crash. Read out of result.data the same way restart_requested is,
    # above: main() is generic across every command, and this keeps the
    # signal out of the command-name dispatch.
    if isinstance(result.data, dict) and result.data.get("preflight_refused"):
        return PREFLIGHT_REFUSAL_EXIT_CODE

    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess test
    # Without this guard, `python -m charlie_work.cli <anything>` imports the
    # module, runs nothing, prints nothing, and exits 0 -- a silent no-op
    # indistinguishable from success (issue #959). That is worse than a
    # traceback for write commands like `tripwire ack`: the operator records a
    # mutation that never happened.
    #
    # SystemExit(main()) rather than a bare main() call, because main() returns
    # a meaningful code and dropping it would reintroduce a quieter version of
    # the same bug. EXIT_RESTART_REQUESTED (3) in particular is a cross-version
    # wire contract read by the supervise-loop wrapper -- a module-form
    # invocation that always exited 0 would make a restart request read as a
    # clean exit, which is the #862 outage.
    raise SystemExit(main())
