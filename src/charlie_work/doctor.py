"""Preflight diagnostics: verify the environment, config, labels, and CI-check
names before a dispatch wave, instead of discovering mismatches mid-run.

The required-check verification derives job names from the consumer repo's
``.github/workflows/*.yml`` at run time — the check list itself stays in
config, but its validity is never asserted by hand.
"""

from __future__ import annotations

import datetime
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import ApiWorkerConfig, OrchestratorConfig
from .fleet_paths import fleet_dir, fleet_dir_virtualization
from .fleet_registry import _load_registry
from .github import (
    GitHubError,
    GitHubLike,
    ISSUE_LIST_FIELDS,
    ISSUE_VIEW_FIELDS,
    LABEL_LIST_FIELDS,
    PR_CHECKS_FIELDS,
    PR_LIST_FIELDS,
    PR_VIEW_FIELDS,
    RECONCILE_ISSUE_FIELDS,
    RECONCILE_PR_FIELDS,
)
from .paths import RuntimePaths
from .prompts import resolve_template
from .runner_slots import (
    ALLOCATION_STATE_FILENAME,
    CLI_ALLOCATION_SOURCE,
    UNATTENDED_ALLOCATION_SOURCE,
    load_allocation_stamp,
)
from .supervise import try_acquire_supervisor_lock


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str
    severity: str = "error"  # "error" | "warning"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "severity": self.severity}


def workflow_job_names(repo_root: Path) -> set[str]:
    """Collect job display names from every GitHub Actions workflow file.

    A job reports its check run under ``jobs.<id>.name`` when set, else the
    job id. Matrix expansions append suffixes GitHub-side, so callers should
    treat these as name prefixes, not exact check-run names.
    """
    names: set[str] = set()
    workflows_dir = repo_root / ".github" / "workflows"
    if not workflows_dir.is_dir():
        return names
    for candidate in sorted(workflows_dir.glob("*.y*ml")):
        try:
            raw = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        jobs = raw.get("jobs") if isinstance(raw, dict) else None
        if not isinstance(jobs, dict):
            continue
        for job_id, job in jobs.items():
            if isinstance(job, dict) and job.get("name"):
                names.add(str(job["name"]))
            else:
                names.add(str(job_id))
    return names


def _check_name_matches(required: str, job_names: set[str]) -> bool:
    if required in job_names:
        return True
    # Matrix jobs report as "<name> (<matrix values>)"; reusable workflows as
    # "<caller> / <callee>". Only these delimited expansions count as a match —
    # a bare prefix must not (job "Test" is not the check "Tests passed").
    for name in job_names:
        if not name:
            continue
        if required.startswith(f"{name} (") or required.startswith(f"{name} / "):
            return True
        if name.startswith(f"{required} (") or name.startswith(f"{required} / "):
            return True
    return False


def _probe_adapter(add: Any, repo_root: Path, config: OrchestratorConfig) -> None:
    """Execute the configured adapter's CLI probe (runs an external binary,
    so only behind --adapter-probe).

    The binary is derived from the configured command template so that a
    custom wrapper set in ``devin.shell_command`` / ``claude_code.command`` is
    exercised — not the hardcoded default name.
    """
    adapter = config.devin.adapter
    if adapter == "devin-shell":
        from .devin_shell import DEFAULT_COMMAND_TEMPLATE, probe_devin

        # Use the operator-configured template; fall back to the package default
        # when empty.  Probe with --version substituted for the dispatch args.
        effective_template = config.devin.shell_command or DEFAULT_COMMAND_TEMPLATE
        binary = effective_template[0]
        probe = probe_devin(repo_root, command=(binary, "--version"))
        add(
            "devin CLI probe",
            probe.ok,
            (probe.stdout.strip() or "ok") if probe.ok else (probe.error or probe.stderr.strip()),
        )
    elif adapter == "claude-code":
        from .claude_code import probe_claude

        # Use the operator-configured command; fall back to the package default
        # when empty.
        _default_claude_binary = "claude"
        effective_command = config.claude_code.command
        binary = effective_command[0] if effective_command else _default_claude_binary
        probe = probe_claude(repo_root, command=(binary, "--version"))
        add(
            "claude CLI probe",
            probe.ok,
            (probe.stdout.strip() or "ok") if probe.ok else (probe.error or probe.stderr.strip()),
        )
    else:
        add(
            "adapter probe",
            True,
            f"adapter `{adapter}` launches nothing itself — no CLI probe applies",
            severity="warning",
        )


def _probe_api_worker(add: Any, paths: RuntimePaths, config: OrchestratorConfig) -> None:
    """Observability probes for the paid ``api`` worker tier (issue #483).

    When the ``api_worker`` section is configured (non-default):

    * ``enabled: true``  — four checks: the active provider's ``api_key_env``
      names a variable present in the environment (the NAME only is reported,
      never any value), ``base_url`` parses as an https URL, the ledger file
      ``<state_dir>/api-budget.json`` is absent-or-parsable, and remaining
      daily/lifetime budget headroom is surfaced from ``api_budget.budget_status``.
    * ``enabled: false`` — a single notice line so a built-but-dormant feature
      stays visible in every doctor run (rollout insurance).

    Near read-only: this probe never settles or writes the ledger itself, but
    ``api_budget.load_ledger`` quarantines a corrupt ledger file (renames it to
    a ``.corrupt-*`` sibling) as a side effect of detecting it. That is the only
    filesystem mutation. Errors surface as check details, never raised.
    """
    # ``configured`` = the section is not the package default. A bare
    # ``api_worker: {enabled: false}`` with no providers/budget is the default
    # and carries nothing to report; a section with providers set but
    # ``enabled: false`` is the built-but-dormant case the notice targets.
    if config.api_worker == ApiWorkerConfig():
        return

    if not config.api_worker.enabled:
        add(
            "api_worker configured but disabled",
            True,
            "api_worker section is configured but disabled (enabled is false) — "
            "flip enabled to true to activate the paid api worker tier",
            severity="warning",
        )
        return

    import os
    from datetime import UTC, datetime

    from .api_budget import budget_status, ledger_path, load_ledger

    provider_name = config.api_worker.provider
    provider = config.api_worker.providers.get(provider_name)
    if provider is None:
        # Config load validates this, but a directly-constructed config must
        # still get a value, not a raise.
        add(
            "api_worker provider",
            False,
            f"active provider {provider_name!r} is not in api_worker.providers",
        )
        return

    # 1. api_key_env present in the environment (NAME only, never the value).
    key_present = bool(os.environ.get(provider.api_key_env))
    add(
        "api_worker api key",
        key_present,
        f"env var {provider.api_key_env!r} is set"
        if key_present
        else f"env var {provider.api_key_env!r} is NOT set — api worker cannot launch",
    )

    # 2. base_url parses as an https URL.
    from urllib.parse import urlparse

    parsed = urlparse(provider.base_url)
    url_ok = parsed.scheme == "https" and bool(parsed.netloc)
    add(
        "api_worker base url",
        url_ok,
        f"{provider.base_url} (https)"
        if url_ok
        else f"base_url {provider.base_url!r} is not a valid https URL",
    )

    # 3. Ledger file absent-or-parsable. load_ledger quarantines a corrupt file
    #    and returns an empty ledger — we detect corruption by checking for a
    #    freshly-created .corrupt sibling. Read-only: no settlement or write.
    ledger_file = ledger_path(paths.root)
    corrupt_before = (
        set(ledger_file.parent.glob(f"{ledger_file.name}.corrupt-*"))
        if ledger_file.parent.exists()
        else set()
    )
    ledger = load_ledger(ledger_file)
    corrupt_after = (
        set(ledger_file.parent.glob(f"{ledger_file.name}.corrupt-*"))
        if ledger_file.parent.exists()
        else set()
    )
    newly_quarantined = corrupt_after - corrupt_before
    if newly_quarantined:
        names = ", ".join(p.name for p in sorted(newly_quarantined))
        add(
            "api_worker budget ledger",
            False,
            f"ledger {ledger_file} was corrupt and quarantined: {names}",
        )
    else:
        add(
            "api_worker budget ledger",
            True,
            f"{ledger_file} ({'present, parsable' if ledger_file.exists() else 'not yet created'})",
        )

    # 4. Remaining daily/lifetime budget headroom.
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    status = budget_status(ledger, config.api_worker.budget, today)
    daily_remaining = max(0.0, config.api_worker.budget.max_usd_per_day - status.spent_today_usd)
    lifetime_remaining = max(
        0.0, config.api_worker.budget.lifetime_usd - status.lifetime_spent_usd
    )
    add(
        "api_worker budget headroom",
        status.daily_headroom and status.lifetime_headroom,
        f"${status.spent_today_usd:.2f} spent today / ${config.api_worker.budget.max_usd_per_day:.2f} cap "
        f"(${daily_remaining:.2f} remaining, {'ok' if status.daily_headroom else 'exhausted'}); "
        f"${status.lifetime_spent_usd:.2f} spent lifetime / ${config.api_worker.budget.lifetime_usd:.2f} cap "
        f"(${lifetime_remaining:.2f} remaining, {'ok' if status.lifetime_headroom else 'exhausted'})",
        severity="warning",
    )


def _surface_sessions(add: Any, repo_root: Path, config: OrchestratorConfig) -> None:
    """Flag launched sessions that failed or whose process died without the
    orchestrator recording an outcome (orphans reconcile cannot see)."""
    from .claude_code import read_worker_records
    from .devin_shell import is_session_alive, read_session_records

    sessions_dir = repo_root / config.devin.sessions_dir
    if not sessions_dir.is_dir():
        add("launched sessions", True, "no sessions directory yet", severity="warning")
        return
    records = [*read_session_records(sessions_dir), *read_worker_records(sessions_dir)]
    failed = [record for record in records if record.error is not None]
    # is_session_alive only reads .pid, so both record kinds duck-type through.
    exited = [
        record for record in records if record.error is None and not is_session_alive(record)
    ]
    detail = f"{len(records)} sidecar record(s): {len(failed)} failed, {len(exited)} exited"
    if failed or exited:
        issues = sorted({record.issue_number for record in [*failed, *exited]})
        detail += f" (issues: {issues}) — check per-session logs in {sessions_dir}"

        # Surface rate-limited deaths specifically
        rate_limited = [r for r in records if getattr(r, "failure_kind", None) == "rate_limited"]
        quota_exhausted = [
            r for r in records if getattr(r, "failure_kind", None) == "quota_exhausted"
        ]
        provider_auth = [r for r in records if getattr(r, "failure_kind", None) == "provider_auth"]
        budget_exceeded = [
            r for r in records if getattr(r, "failure_kind", None) == "budget_exceeded"
        ]
        if rate_limited:
            rl_issues = sorted({r.issue_number for r in rate_limited})
            detail += f" | rate-limited: {rl_issues}"
        if quota_exhausted:
            qe_issues = sorted({r.issue_number for r in quota_exhausted})
            detail += f" | quota-exhausted: {qe_issues}"
        if provider_auth:
            pa_issues = sorted({r.issue_number for r in provider_auth})
            detail += f" | provider-auth: {pa_issues}"
        if budget_exceeded:
            be_issues = sorted({r.issue_number for r in budget_exceeded})
            detail += f" | budget-exceeded: {be_issues}"

    add("launched sessions", not failed, detail, severity="warning")

    if failed or exited:
        _surface_post_mortems(add, repo_root, sessions_dir, [*failed, *exited])


def _surface_post_mortems(
    add: Any, repo_root: Path, sessions_dir: Path, dead_records: list[Any]
) -> None:
    """Issue #261: for each dead session, surface its post-mortem terminal
    cause (from the Devin CLI session store, when extraction succeeded) and
    any preserved attempt refs (unpushed commits salvaged before a redispatch
    reset the branch) — both invisible in the plain session-record summary
    above.

    Best-effort/read-only: a missing or unreadable post-mortem sidecar for a
    given issue is silently skipped (worker_blocked detection is opportunistic,
    not guaranteed), never treated as a doctor failure.
    """
    from .attempt_refs import list_attempt_refs
    from .post_mortem import read_post_mortem

    issue_numbers = sorted({record.issue_number for record in dead_records})
    lines: list[str] = []
    for issue_number in issue_numbers:
        post_mortem = read_post_mortem(sessions_dir, issue_number)
        attempt_refs = list_attempt_refs(repo_root, issue_number)
        if post_mortem is None and not attempt_refs:
            continue
        parts = [f"issue #{issue_number}"]
        if post_mortem is not None:
            if post_mortem.failure_kind:
                parts.append(f"failure_kind={post_mortem.failure_kind}")
            if post_mortem.terminal_tool:
                parts.append(f"terminal_tool={post_mortem.terminal_tool}")
            if post_mortem.terminal_reason:
                reason = post_mortem.terminal_reason.strip().splitlines()[0][:120]
                parts.append(f"reason={reason!r}")
            if not post_mortem.matched and post_mortem.extraction_error:
                parts.append(f"extraction_error={post_mortem.extraction_error}")
        if attempt_refs:
            parts.append(f"attempt_refs={list(attempt_refs)}")
        lines.append(" ".join(parts))

    if lines:
        add(
            "dead session post-mortems",
            True,
            "; ".join(lines),
            severity="warning",
        )


def _allocation_writer_label(source: str | None) -> str:
    """Describe which path wrote an allocation state file, for probe output.

    An unrecognised value is echoed rather than collapsed into "unknown": if a
    future writer forgets to extend ``AllocationSource``, the probe should name
    what it actually found instead of hiding it.
    """
    if source is None:
        return "writer unrecorded — file predates provenance tracking"
    if source == UNATTENDED_ALLOCATION_SOURCE:
        return "unattended fleet pass"
    if source == CLI_ALLOCATION_SOURCE:
        return "a manual `charlie runners allocate`"
    return f"an unrecognised writer {source!r}"


def _check_runner_allocation(
    add: Any, config: OrchestratorConfig, fleet_dir_override: str | None = None
) -> None:
    """Report whether host-wide runner allocation is actually running (issue #590).

    Every way the allocation prologue can decline to act is silent by nature: a
    false ``enabled`` flag, a config object built by code that predates the
    section, or a registry with no reachable repo root all make it return without
    doing anything — and a converged host looks exactly like one where allocation
    never ran at all. Logs cannot settle it either: the daemon's stderr has proven
    lossy in practice, so the absence of a log line is not evidence.

    ``run_allocation_pass`` rewrites ``runner-allocation.json`` on every non-dry
    pass even when no slot moves, which makes that file's ``updated_at`` the only
    positive evidence that a pass happened.

    Age alone is not enough, though. ``charlie runners allocate`` writes the same
    host-wide file, and CLAUDE.md *requires* post-reboot procedures to call exactly
    that — so an operator's manual run would otherwise make this probe read healthy
    for three intervals, during the very window someone is diagnosing #590. Each
    pass records which path wrote it and only the unattended one is accepted as
    evidence here; a manual write is reported as "cannot confirm" rather than
    "fine", because the file keeps just the latest write.
    """
    allocation = getattr(config, "runner_allocation", None)
    if allocation is None or not allocation.enabled:
        return

    state_dir = fleet_dir(override=fleet_dir_override)
    interval = max(config.supervisor.full_pass_interval_seconds, 1)
    # Three intervals: one missed pass is normal jitter (a pass can run long), a
    # sustained gap is not.
    stale_after = interval * 3
    budget = allocation.max_running_runners

    stamp = load_allocation_stamp(state_dir)
    if stamp is None:
        add(
            "runner allocation",
            False,
            f"enabled (budget {budget}) but has never run: "
            f"{state_dir / ALLOCATION_STATE_FILENAME} absent, "
            f"expected a pass every {interval}s",
            severity="warning",
        )
        return

    if stamp.updated_at is None:
        add(
            "runner allocation",
            False,
            f"enabled but {ALLOCATION_STATE_FILENAME} has no readable updated_at stamp",
            severity="warning",
        )
        return

    # Clock skew or a hand-edited stamp can date the write in the future. A
    # negative age is not freshness evidence, so clamp it instead of reporting
    # "last pass -42s ago" as healthy.
    now = datetime.datetime.now(datetime.timezone.utc)
    age = max(0, int((now - stamp.updated_at).total_seconds()))
    writer = _allocation_writer_label(stamp.source)

    if age > stale_after:
        add(
            "runner allocation",
            False,
            f"enabled (budget {budget}) but the last pass ({writer}) was {age}s ago, "
            f"over the {stale_after}s staleness bound — allocation is configured but "
            f"is not running unattended (issue #590)",
            severity="warning",
        )
        return

    if stamp.source != UNATTENDED_ALLOCATION_SOURCE:
        add(
            "runner allocation",
            False,
            f"enabled (budget {budget}) but the most recent pass {age}s ago was "
            f"{writer}, which overwrites the same file the unattended pass uses — "
            f"this cannot confirm the daemon is rebalancing (issue #590)",
            severity="warning",
        )
        return

    add(
        "runner allocation",
        True,
        f"last unattended pass {age}s ago, budget {budget}",
    )


def _check_fleet_dir_virtualization(add: Any, fleet_dir_override: str | None = None) -> None:
    """Warn when the fleet directory is per-process virtualized (issue #624).

    MSIX/container copy-on-write redirection makes the literal fleet-dir path
    string identical in both the container and the host while naming different
    files: reads pass through to the real file, but the first write forks a
    private copy that daemons reading the same path string never see. This
    cost a full day on #590 — ``runner_allocation`` was believed deployed and
    enabled since 09:24, while the file the fleet supervisor actually read
    never had the section at all.

    The signal is that the literal path and its resolved form disagree — never
    a hardcoded package moniker (which would rot on the next app update, only
    cover one container, and violate the no-hardcoded-lists rule). A
    virtualized fleet dir is not fatal for an interactive human running
    ``charlie doctor`` from a packaged terminal — reads still pass through
    until something writes — so this is a warning, not an error. It names both
    paths and states plainly that any write forks a private copy daemons will
    never see, referencing #590 for the failure it produced.

    Repo-agnostic by construction: the fleet dir is a host-wide per-process
    property, so the same probe fires regardless of which registered repo the
    operator ran ``charlie doctor`` from.
    """
    diverged = fleet_dir_virtualization(override=fleet_dir_override)
    if diverged is None:
        return
    literal, resolved = diverged
    add(
        "fleet dir virtualization",
        False,
        f"fleet dir {literal} resolves to {resolved} — host-wide state written "
        f"here is invisible to scheduled tasks and daemons reading the same "
        f"path string: the first write forks a private copy they will never "
        f"see (this is the exact shape of the #590 failure). Daemon-visible "
        f"state must be written via a non-redirected route (e.g. a UNC path).",
        severity="warning",
    )


def _check_fleet_supervisor(add: Any, fleet_dir_override: str | None = None) -> None:
    """Warn when a fleet registry exists but no supervisor appears to be driving it.

    The check is per-repo aware: if the fleet supervisor lock is not held, each
    repo's per-repo supervisor lock is checked individually. A single supervised
    repo no longer hides an unsupervised one.
    """
    fleet_json_path = fleet_dir(override=fleet_dir_override) / "fleet.json"
    if not fleet_json_path.exists():
        return
    registry = _load_registry(fleet_json_path)
    repos = registry.get("repos", {})
    if not repos:
        return

    fleet_lock_path = fleet_dir(override=fleet_dir_override) / "fleet-supervisor.lock"
    if fleet_lock_path.exists():
        fleet_lock = try_acquire_supervisor_lock(fleet_lock_path)
        if fleet_lock is None:
            add("fleet supervisor", True, "fleet supervisor appears to be running")
            return
        fleet_lock.release()

    supervised_repo_keys: list[str] = []
    unsupervised_repo_keys: list[str] = []
    unreachable_repo_keys: list[str] = []
    for repo_key, entry in repos.items():
        state_dir_str = entry.get("state_dir")
        if not state_dir_str:
            unsupervised_repo_keys.append(repo_key)
            continue
        state_dir = Path(state_dir_str)
        if not state_dir.exists():
            unreachable_repo_keys.append(repo_key)
            continue
        repo_lock_path = state_dir / "supervisor.lock"
        if repo_lock_path.exists():
            repo_lock = try_acquire_supervisor_lock(repo_lock_path)
            if repo_lock is None:
                supervised_repo_keys.append(repo_key)
            else:
                repo_lock.release()
                unsupervised_repo_keys.append(repo_key)
        else:
            unsupervised_repo_keys.append(repo_key)

    parts: list[str] = []
    if supervised_repo_keys:
        parts.append(f"supervised={len(supervised_repo_keys)} ({', '.join(supervised_repo_keys)})")
    if unsupervised_repo_keys:
        parts.append(
            f"unsupervised={len(unsupervised_repo_keys)} ({', '.join(unsupervised_repo_keys)})"
        )
    if unreachable_repo_keys:
        parts.append(
            f"unreachable={len(unreachable_repo_keys)} ({', '.join(unreachable_repo_keys)})"
        )

    if not unsupervised_repo_keys:
        detail = "fleet supervisor appears to be running"
        if supervised_repo_keys:
            detail += f" for all {', '.join(supervised_repo_keys)}"
        if unreachable_repo_keys:
            detail += f"; {len(unreachable_repo_keys)} repo(s) have no reachable state_dir"
        add("fleet supervisor", True, detail)
        return

    detail = (
        f"{len(repos)} repo(s) registered in fleet.json; "
        "fleet supervisor not running; " + ", ".join(parts)
    )
    if supervised_repo_keys:
        detail += (
            "; run `charlie fleet supervise` for continuous operation or schedule "
            "`charlie fleet bash-rats` for the unsupervised repo(s)"
        )
    else:
        detail += (
            "; run `charlie fleet supervise` for continuous operation or schedule "
            "`charlie fleet bash-rats`"
        )

    add(
        "fleet supervisor",
        False,
        detail,
        severity="warning",
    )


def _validate_gh_field_lists(add: Any, gh: GitHubLike) -> None:
    """Validate gh --json field lists against the live gh CLI.

    Executes each field list as a read-only query with --limit 1 and reports
    any invalid/unknown fields with the gh error text. This catches contract
    drift between the hardcoded field lists and the actual gh CLI schema.
    """

    # Discover probe targets dynamically instead of hardcoding item #1
    def _find_pr_number() -> int | None:
        """Find a real PR number to probe, or None if no PRs exist."""
        try:
            result = gh.run(
                ["pr", "list", "--state", "all", "--limit", "1", "--json", "number"],
                json_output=True,
            )
            if result and isinstance(result, list) and result:
                return result[0].get("number")
        except GitHubError:
            pass
        return None

    def _find_issue_number() -> int | None:
        """Find a real issue number to probe, or None if no issues exist."""
        try:
            result = gh.run(
                ["issue", "list", "--state", "all", "--limit", "1", "--json", "number"],
                json_output=True,
            )
            if result and isinstance(result, list) and result:
                return result[0].get("number")
        except GitHubError:
            pass
        return None

    pr_number = _find_pr_number()
    issue_number = _find_issue_number()

    # Map of field list name to (command, fields) tuples
    # Commands that need specific item numbers use placeholders
    field_lists = {
        "ISSUE_LIST_FIELDS": (
            ["issue", "list", "--state", "open", "--limit", "1"],
            ISSUE_LIST_FIELDS,
        ),
        "ISSUE_VIEW_FIELDS": (
            ["issue", "view", str(issue_number)] if issue_number else None,
            ISSUE_VIEW_FIELDS,
        ),
        "PR_LIST_FIELDS": (["pr", "list", "--state", "open", "--limit", "1"], PR_LIST_FIELDS),
        "PR_VIEW_FIELDS": (
            ["pr", "view", str(pr_number)] if pr_number else None,
            PR_VIEW_FIELDS,
        ),
        "PR_CHECKS_FIELDS": (
            ["pr", "checks", str(pr_number)] if pr_number else None,
            PR_CHECKS_FIELDS,
        ),
        "LABEL_LIST_FIELDS": (["label", "list", "--limit", "1"], LABEL_LIST_FIELDS),
        "RECONCILE_PR_FIELDS": (
            ["pr", "list", "--state", "all", "--limit", "1"],
            RECONCILE_PR_FIELDS,
        ),
        "RECONCILE_ISSUE_FIELDS": (
            ["issue", "list", "--state", "open", "--limit", "1"],
            RECONCILE_ISSUE_FIELDS,
        ),
    }

    for list_name, (base_cmd, fields) in field_lists.items():
        # Skip if no probe target available
        if base_cmd is None:
            if list_name in ("ISSUE_VIEW_FIELDS",):
                add(
                    f"gh field list: {list_name}",
                    True,
                    "skipped (no issue available to probe)",
                    severity="warning",
                )
            elif list_name in ("PR_VIEW_FIELDS", "PR_CHECKS_FIELDS"):
                add(
                    f"gh field list: {list_name}",
                    True,
                    "skipped (no PR available to probe)",
                    severity="warning",
                )
            continue

        cmd = [*base_cmd, "--json", fields]
        try:
            gh.run(cmd, json_output=True)
            add(f"gh field list: {list_name}", True, f"valid ({len(fields.split(','))} fields)")
        except GitHubError as exc:
            error_msg = str(exc)
            # Classify errors: only actual field errors get the "invalid field(s)" label
            # Field errors have a specific shape: "Unknown JSON field: ..." or "invalid JSON field: ..."
            is_field_error = any(
                phrase in error_msg
                for phrase in ("Unknown JSON field:", "invalid JSON field:", "invalid field")
            )

            # Special case: gh pr checks fails with non-zero exit when no CI is configured
            # This is not a field error - it's a missing feature
            if list_name == "PR_CHECKS_FIELDS" and "no checks reported" in error_msg.lower():
                add(
                    f"gh field list: {list_name}",
                    True,
                    "skipped (no CI configured on probe PR)",
                    severity="warning",
                )
            elif is_field_error:
                add(
                    f"gh field list: {list_name}",
                    False,
                    f"invalid field(s): {error_msg}",
                )
            else:
                add(
                    f"gh field list: {list_name}",
                    False,
                    f"probe failed (not a field error): {error_msg}",
                )


def run_doctor(
    repo_root: Path,
    paths: RuntimePaths,
    config: OrchestratorConfig,
    config_path: Path | None,
    gh: GitHubLike,
    *,
    adapter_probe: bool = False,
    live: bool = False,
    fleet_dir_override: str | None = None,
) -> tuple[bool, list[DoctorCheck]]:
    checks: list[DoctorCheck] = []

    def add(name: str, ok: bool, detail: str, *, severity: str = "error") -> None:
        checks.append(DoctorCheck(name=name, ok=ok, detail=detail, severity=severity))

    # -- environment ---------------------------------------------------------
    gh_path = shutil.which("gh")
    add("gh on PATH", gh_path is not None, gh_path or "GitHub CLI `gh` not found on PATH")
    if gh_path:
        try:
            gh.run(["auth", "status"])
            add("gh auth", True, "authenticated")
        except GitHubError as exc:
            add("gh auth", False, str(exc))

    # -- config --------------------------------------------------------------
    if config_path is None:
        add(
            "config file",
            False,
            "no orchestrator.config.yaml found at the repo root — running on package "
            "defaults (required_checks is empty; see examples/)",
            severity="warning",
        )
    else:
        add("config file", True, str(config_path))

    if config.auto_merge.enabled and not config.auto_merge.required_checks:
        add(
            "required checks configured",
            False,
            "auto_merge.enabled is true but required_checks is empty — merge-ready "
            "would gate on the review decision alone",
        )
    else:
        add(
            "required checks configured",
            True,
            f"{len(config.auto_merge.required_checks)} required check(s)",
        )

    # -- required checks vs live workflow files ------------------------------
    job_names = workflow_job_names(repo_root)
    if config.auto_merge.required_checks:
        if job_names:
            for required in config.auto_merge.required_checks:
                matched = _check_name_matches(required, job_names)
                add(
                    f"check name: {required}",
                    matched,
                    "matches a workflow job"
                    if matched
                    else f"no job in .github/workflows/ reports this name; found: {sorted(job_names)}",
                )
        else:
            add(
                "workflow files",
                False,
                "required_checks configured but no parseable .github/workflows/*.yml found",
                severity="warning",
            )

    # -- labels --------------------------------------------------------------
    try:
        live_labels = {
            str(item.get("name") or "") for item in gh.label_list() if isinstance(item, dict)
        }
        missing = [label for label in config.labels.all if label not in live_labels]
        add(
            "github labels",
            not missing,
            "all orchestration labels exist"
            if not missing
            else f"missing labels {missing} — run `bootstrap-labels`",
        )
    except GitHubError as exc:
        add("github labels", False, f"could not list labels: {exc}", severity="warning")

    # -- state ---------------------------------------------------------------
    # Read-only preflight: parse the raw bytes with json.loads so we never
    # trigger load_state's quarantine side-effect (which renames state.json to
    # state.json.corrupt-*).  A missing file is fine (first run).
    if not paths.state_file.exists():
        add("state file", True, f"{paths.state_file} (not yet created)")
    else:
        try:
            raw_state = json.loads(paths.state_file.read_text(encoding="utf-8"))
            if not isinstance(raw_state, dict):
                raise ValueError("state file is not a JSON object")
            add(
                "state file",
                True,
                f"{paths.state_file} (issues: {len(raw_state.get('issues', {}))}, "
                f"prs: {len(raw_state.get('prs', {}))})",
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            add("state file", False, f"{paths.state_file}: {exc}")

    # Surface any previously-quarantined corrupt state files so the operator
    # knows to inspect or clean them up.
    corrupt_files = sorted(paths.state_file.parent.glob(f"{paths.state_file.name}.corrupt-*"))
    if corrupt_files:
        names = ", ".join(p.name for p in corrupt_files)
        add(
            "state file quarantine",
            False,
            f"{len(corrupt_files)} quarantined corrupt state file(s) in "
            f"{paths.state_file.parent}: {names}",
            severity="warning",
        )

    # -- adapters ------------------------------------------------------------
    if config.devin.adapter == "command" and not config.devin.dispatch_command:
        add("dispatch adapter", False, "adapter is `command` but dispatch_command is empty")
    else:
        add("dispatch adapter", True, config.devin.adapter)

    # Report the effective devin-shell worker model so the operator can see
    # what dispatch will actually launch.
    if config.devin.adapter == "devin-shell":
        if config.devin.worker_model:
            add(
                "devin-shell worker model",
                True,
                f"config-driven: {config.devin.worker_model}",
            )
        else:
            add(
                "devin-shell worker model",
                True,
                "CLI default (no devin.worker_model configured)",
                severity="warning",
            )

    # claude-code worktrees junction a shared venv in; surface a missing
    # venv_source at preflight rather than deferring it to the first dispatch.
    if config.devin.adapter == "claude-code" and config.claude_code.venv_source:
        venv = Path(config.claude_code.venv_source)
        if not venv.is_absolute():
            venv = repo_root / venv
        add(
            "claude-code venv source",
            venv.is_dir(),
            str(venv)
            if venv.is_dir()
            else f"claude_code.venv_source does not exist: {venv} "
            "(set it to null to disable venv sharing)",
        )

    # -- api worker observability (issue #483) ------------------------------
    # Always runs (not gated on --adapter-probe): these are config/environment
    # checks, not external CLI probes. Read-only — never mutates the ledger.
    _probe_api_worker(add, paths, config)

    if adapter_probe:
        _probe_adapter(add, repo_root, config)
        _surface_sessions(add, repo_root, config)

    if live:
        _validate_gh_field_lists(add, gh)

    if config.cross_family.enabled:
        command = config.cross_family.command
        binary = command.split()[0] if isinstance(command, str) else str(command[0])
        found = shutil.which(binary)
        add(
            "cross-family binary",
            found is not None,
            found or f"cross_family.enabled but `{binary}` not found on PATH",
            severity="warning",
        )

    # -- prompts -------------------------------------------------------------
    prompts_dir = config.runtime.prompts_dir
    search_dirs: tuple[Path, ...] = ()
    if prompts_dir:
        override = Path(prompts_dir)
        if not override.is_absolute():
            override = repo_root / override
        if override.is_dir():
            search_dirs = (override,)
            add("prompts dir", True, str(override))
        else:
            add("prompts dir", False, f"runtime.prompts_dir does not exist: {override}")
    template = config.dispatch.worker_template
    template_path = resolve_template(template, search_dirs)
    add(
        f"worker template: {template}",
        template_path.is_file(),
        str(template_path) if template_path.is_file() else f"not found: {template_path}",
    )

    # -- fleet supervisor ----------------------------------------------------
    _check_fleet_supervisor(add, fleet_dir_override=fleet_dir_override)

    # -- fleet dir virtualization (issue #624) -------------------------------
    # Read-only: compares the literal fleet-dir path against its resolved form.
    # Never raises and never writes; a virtualized fleet dir is a warning, not
    # an error, because reads still pass through until something writes.
    _check_fleet_dir_virtualization(add, fleet_dir_override=fleet_dir_override)

    # -- host-wide runner allocation (issue #590) ----------------------------
    # Read-only: compares the allocation state file's age against the pass
    # interval. Never starts, parks, or plans anything.
    _check_runner_allocation(add, config, fleet_dir_override=fleet_dir_override)

    hard_failures = [check for check in checks if not check.ok and check.severity == "error"]
    return (not hard_failures, checks)
