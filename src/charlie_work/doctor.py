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
from .doctor_local_backend import _check_local_issue_backend
from .env_sanitize import worker_github_token_findings
from .fleet_paths import fleet_dir, fleet_dir_virtualization
from .fleet_registry import _load_registry
from . import layout
from .instrumentation import _db_path, query_events
from .github import (
    GitHubError,
    GitHubLike,
    ISSUE_LIST_FIELDS,
    ISSUE_VIEW_FIELDS,
    LABEL_LIST_FIELDS,
    PR_CHECKS_FIELDS,
    PR_LIST_FIELDS,
    PR_VIEW_FIELDS,
    PROBE_NUMBER_FIELDS,
    RECONCILE_ISSUE_FIELDS,
    RECONCILE_PR_FIELDS,
)
from .local_work_park import publishes_pull_requests
from .paths import RuntimePaths, resolved_layout
from .prompts import resolve_template
from ci_fleet.charlie_work_adapter import (
    ALLOCATION_STATE_FILENAME,
    CLI_ALLOCATION_SOURCE,
    UNATTENDED_ALLOCATION_SOURCE,
    load_allocation_stamp,
)
from ci_fleet.provenance import REFUSAL_STATE_FILENAME, load_refusal_streak
from .supervise import _self_deploy_state_path, orchestrator_root, try_acquire_supervisor_lock


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


def workflow_job_matrix_flags(repo_root: Path) -> dict[str, bool]:
    """Map each workflow job display name to whether that job uses ``strategy.matrix``.

    Derives matrix-ness from the parsed workflow YAML rather than a hardcoded
    job-name list, scoped PER JOB (not repo-wide), so the required-check
    verifier (issue #1508) can decide whether matrix-suffix tolerance
    (``Name (suffix)`` matching job ``Name``) is justified by an actual matrix
    expansion ON THE MATCHED JOB -- or is matching a stale suffixed
    required-check entry that the exact-match merge gate (``checks.py``) would
    report ``missing`` forever.

    A previous repo-wide boolean could justify a tolerance match against a
    non-matrix job in workflow B merely because an unrelated job in workflow A
    had a matrix -- silently reintroducing the exact false-pass hazard #1508
    closed once any workflow in a multi-workflow repo gained a matrix job.
    Scoping to the matched job's own matrix flag is the single point of
    enforcement.

    A display name shared by several jobs is matrix-backed if ANY job reporting
    under that name has a matrix (OR-combined), so a tolerance match on that
    name is justified whenever at least one underlying job would actually
    expand it.
    """
    flags: dict[str, bool] = {}
    workflows_dir = repo_root / ".github" / "workflows"
    if not workflows_dir.is_dir():
        return flags
    for candidate in sorted(workflows_dir.glob("*.y*ml")):
        try:
            raw = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        jobs = raw.get("jobs") if isinstance(raw, dict) else None
        if not isinstance(jobs, dict):
            continue
        for job_id, job in jobs.items():
            if not isinstance(job, dict):
                continue
            # Mirror workflow_job_names' name resolution exactly so the keys
            # here align with the names a tolerance match is computed against.
            name = str(job["name"]) if job.get("name") else str(job_id)
            strategy = job.get("strategy")
            has_matrix = isinstance(strategy, dict) and isinstance(strategy.get("matrix"), dict)
            flags[name] = flags.get(name, False) or has_matrix
    return flags


def _tolerance_match_base_names(required: str, job_names: set[str]) -> list[str]:
    """Return the workflow job display names a required check matches via tolerance.

    A tolerance match is a matrix-suffix or reusable-workflow-delimiter match
    (``Name (suffix)`` / ``caller / callee``). The returned names are the
    workflow job display names (the ``name``/``id`` from
    :func:`workflow_job_names`) that the suffix/delimiter expands from -- i.e.
    the jobs whose own ``strategy.matrix`` flag decides whether the tolerance
    is justified. A bare prefix never counts (job "Test" is not the check
    "Tests passed").
    """
    bases: list[str] = []
    for name in job_names:
        if not name:
            continue
        if required.startswith(f"{name} (") or required.startswith(f"{name} / "):
            bases.append(name)
        elif name.startswith(f"{required} (") or name.startswith(f"{required} / "):
            bases.append(name)
    return bases


def _check_name_match_kind(required: str, job_names: set[str]) -> str | None:
    """Classify how a required check matches workflow job names.

    Returns ``"exact"`` for a direct name match, ``"tolerance"`` for a
    matrix-suffix or reusable-workflow-delimiter match
    (``Name (suffix)`` / ``caller / callee``), or ``None`` for no match.

    The merge gate in ``checks.py`` uses EXACT match only, so a
    ``"tolerance"``-only match is a stale-required-check-entry hazard when the
    matched job has no ``strategy.matrix`` (issue #1508): doctor would pass a
    ``Tests (windows-latest)`` required-check entry that the merge gate
    reports ``missing`` forever. Whether the tolerance is justified is decided
    by the caller, scoped to the matched job via
    :func:`workflow_job_matrix_flags` + :func:`_tolerance_match_base_names`.
    """
    if required in job_names:
        return "exact"
    if _tolerance_match_base_names(required, job_names):
        return "tolerance"
    return None


def _probe_adapter(add: Any, repo_root: Path, config: OrchestratorConfig) -> None:
    """Execute the configured adapter's CLI probe (runs an external binary,
    so only behind --adapter-probe).

    The binary is derived from the configured command template so that a
    custom wrapper set in ``devin.shell_command`` / ``claude_code.command`` is
    exercised — not the hardcoded default name.
    """
    adapter = config.worker.harness
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


#  Issue #873 Part 2 / #1001: the token variable names and the adapter-path
#  predicate live in env_sanitize.py now — shared by this preflight check and
#  the dispatch gate in workflow.py so they cannot drift (issue #1001). The
#  pre-#1001 private _STRIPPED_GH_TOKEN_VARS copy was deleted when
#  worker_github_token_findings became the single predicate.


def _check_worker_github_token(add: Any, config: OrchestratorConfig) -> None:
    """Flag a dispatch-enabled adapter with no scoped GitHub token configured
    for its workers (issue #873 Part 2).

    Delegates to :func:`env_sanitize.worker_github_token_findings` — the single
    predicate shared with the dispatch gate in ``workflow.py:_dispatch_impl``
    (issue #1001) — so the doctor preflight and the dispatch gate cannot
    disagree about whether the fleet is healthy.

    Background: ``env_sanitize.sanitize_env`` (issue #502) is a deliberate
    security control — it strips ``GH_TOKEN``/``GITHUB_TOKEN`` (and the GHES
    equivalents) from every worker subprocess's environment and points
    ``GH_CONFIG_DIR`` at an empty, worktree-local directory, so a worker can
    never use the orchestrator's own ``gh`` credentials. The *only* sanctioned
    way for a worker to reach ``gh`` is an operator-supplied token in
    ``devin.worker_env``/``claude_code.worker_env``, which ``launch_devin_session``
    (devin_shell.py) and ``launch_claude_worker`` (claude_code.py) each merge
    back in AFTER ``sanitize_env()`` runs — see the "Merge order" comments on
    ``DevinConfig.worker_env``/``ClaudeCodeConfig.worker_env`` in config.py.

    Without an operator-configured token, a dispatched worker has no
    sanctioned credential and either stalls waiting on a human, or — the
    porousness issue #873 also names — improvises its way to a locally cached
    Git Credential Manager entry that ``sanitize_env`` does not (and, per that
    issue, deliberately does not yet) neutralize. Both outcomes are silent
    until this check: nothing about a missing token fails loudly before
    dispatch today.

    This check reads only ``config.devin.worker_env`` /
    ``config.claude_code.worker_env`` — it never reads the process
    environment and never calls ``sanitize_env`` — so it cannot report a
    false-healthy result from an ambient ``GH_TOKEN`` the sanitizer would
    strip anyway, and it cannot widen what ``sanitize_env`` passes through;
    it only observes whether the sanctioned provisioning path (the config
    ``worker_env`` mapping) has been used. It reports presence as a boolean
    only — it never logs a token value or any prefix of one.

    Only fires for the adapter families that actually route through
    ``sanitize_env``'s merge: ``devin-shell`` (sources ``devin.worker_env``)
    and ``claude-code``/``api`` (both source ``claude_code.worker_env`` — the
    ``api`` adapter reuses the claude-code launch path, see
    ``workflow.py:_adapter_settings``). ``manual`` only writes a session
    manifest for a human to act on and never launches a worker subprocess;
    ``command`` runs ``subprocess_runner.run_captured`` with no ``sanitize_env``
    call at all, so it inherits the orchestrator's full (unsanitized)
    environment. Neither has the failure mode this check targets.

    Two other paths dispatch through the claude-code launch path
    (``claude_code.worker_env``) regardless of the configured default
    ``worker.harness``, so a ``devin-shell``/``manual``/``command`` default
    can still stall a worker mid-pass with no visible finding unless both are
    covered:

    * ``config.api_worker.enabled`` being ``True`` enables the ``api``
      adapter tier, which reuses the claude-code launch path
      (``claude_code.worker_env``) — so a missing ``claude_code`` token
      can stall an api-adapter worker even when the default
      ``worker.harness`` is not claude-code.
    * ``_rescue_adapter_settings`` (workflow.py) *always* forces
      ``adapter="claude-code"`` for the bounded rescue tier (issue #555)
      once ``config.rescue.enabled`` is true, independent of
      ``api_worker.enabled`` — rescue and the paid api tier are unrelated
      toggles, so checking one does not imply the other is off.

    Either toggle alone is enough for a dispatched worker to hit the
    claude-code path, so this fires as a second, separately-named finding
    whenever *either* is true, so that combination isn't hidden behind the
    primary adapter's check.

    Severity is ``warning``, not the default ``error``: issue #873 is
    explicit that the sanctioned fix (an operator configuring a scoped token)
    is a deferred, human action this check only surfaces — not one this
    check performs or can force. An ``error`` severity would make
    ``run_doctor``'s overall ``ok`` unconditionally ``False`` on every
    production config that hasn't yet been given a token, with no path to
    green except that same deferred human step (see ``cli.py``'s
    severity-independent ``failed`` list — the finding is exactly as visible
    at ``warning``, it just doesn't block).
    """
    for finding in worker_github_token_findings(config):
        add(
            finding.name,
            finding.ok,
            finding.detail,
            severity="warning",
        )


def _probe_api_worker(
    add: Any,
    paths: RuntimePaths,
    config: OrchestratorConfig,
    *,
    now: datetime.datetime | None = None,
) -> None:
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

    ``now`` is the injectable clock used to derive the ledger's ``today`` key
    (issue #828): defaults to ``datetime.now(UTC)`` when not supplied, so
    production behavior is byte-identical. ``run_doctor`` samples one ``now``
    per doctor run and passes it here and to ``_check_runner_allocation``.
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
    resolved_now = now if now is not None else datetime.now(UTC)
    today = resolved_now.strftime("%Y-%m-%d")
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

    sessions_dir = resolved_layout(config, repo_root).sessions_dir
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


def _surface_in_progress_corroboration(
    add: Any, repo_root: Path, config: OrchestratorConfig, *, now: datetime.datetime | None
) -> None:
    """Issue #1346: surface the watchdog's corroboration probe for in-progress
    workers so an alive-but-polling worker (stale sidecar log mtime, fresh
    sessions.db / per-PID log / events.jsonl) is visibly distinguishable from a
    genuinely stalled one in doctor output.

    Read-only and cheap: reuses ``iter_workers`` +
    ``real_activity_probe_for`` + ``classify_worker_health`` -- the exact same
    code path the stall watchdog consults -- so the displayed classification
    and the displayed corroboration can never diverge. Never raises: a missing
    sessions directory or a probe I/O failure is reported as "no in-progress
    workers" rather than crashing doctor.

    The check is informational (``ok=True``, ``severity="warning"``) unless a
    genuinely stalled worker is found, in which case ``ok=False`` so the
    operator sees it surfaced alongside the existing ``launched sessions``
    check rather than having to infer staleness from a silent log mtime.
    """
    from datetime import UTC

    from .worker import classify_worker_health, iter_workers, real_activity_probe_for

    sessions_dir = resolved_layout(config, repo_root).sessions_dir
    if not sessions_dir.is_dir():
        return

    resolved_now = now if now is not None else datetime.datetime.now(datetime.timezone.utc)
    dt_now = resolved_now if resolved_now.tzinfo is not None else resolved_now.replace(tzinfo=UTC)

    alive_polling: list[str] = []
    stalled: list[str] = []
    healthy_fresh_log: list[str] = []
    for view in iter_workers(sessions_dir):
        if not view.is_alive():
            continue
        probe = real_activity_probe_for(view, config, dt_now)
        health = classify_worker_health(view, config, dt_now, probe)
        latest_ts = probe.latest_timestamp
        latest_iso = latest_ts.isoformat() if latest_ts is not None else "none"
        latest_src = probe.latest_source or "no source"
        fresh = probe.is_fresh(config.watchdog.stall_minutes)
        # Sidecar log staleness: view.last_activity_at is the log mtime ISO.
        log_stale = _log_mtime_is_stale(view, config.watchdog.stall_minutes, dt_now)
        tag = f"issue #{view.issue_number} (pid={view.pid}, health={health.value}, "
        tag += f"corroboration={latest_iso} via {latest_src}, fresh={fresh})"
        if health.value in ("stalled", "dead"):
            stalled.append(tag)
        elif log_stale and fresh:
            # The exact #1346 shape: sidecar log frozen, real session still
            # moving. Healthy per the watchdog, but indistinguishable from
            # stalled on log mtime alone -- surface it explicitly.
            alive_polling.append(tag)
        else:
            healthy_fresh_log.append(tag)

    parts: list[str] = []
    if alive_polling:
        parts.append(f"alive-but-polling ({len(alive_polling)}): " + "; ".join(alive_polling))
    if stalled:
        parts.append(f"stalled/dead ({len(stalled)}): " + "; ".join(stalled))
    if healthy_fresh_log:
        parts.append(f"healthy ({len(healthy_fresh_log)})")
    detail = "; ".join(parts) if parts else f"no in-progress workers in {sessions_dir}"

    # ok=True unless a genuinely stalled/dead worker is present -- those are
    # real problems. Alive-but-polling is informational (the watchdog already
    # defers them), surfaced so an operator does not chase them as stalls.
    add(
        "in-progress worker corroboration",
        not stalled,
        detail,
        severity="warning",
    )


def _log_mtime_is_stale(view: Any, stall_minutes: int, now: datetime.datetime) -> bool:
    """Return True when the worker's sidecar log mtime is older than
    ``stall_minutes``. Stats the log file directly (via ``view.log_stat()``)
    rather than reading the sidecar's stored ``last_activity_at`` field, so
    the verdict reflects the *current* log mtime -- the same signal a
    log-mtime monitor reads. Best-effort: a missing/unreadable log is treated
    as not-stale (the corroboration probe is the authoritative signal there)."""
    from datetime import timedelta

    stat_result = view.log_stat()
    if stat_result is None:
        return False
    log_mtime = datetime.datetime.fromtimestamp(stat_result.st_mtime, tz=datetime.timezone.utc)
    return (now - log_mtime) > timedelta(minutes=stall_minutes)


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
    add: Any,
    config: OrchestratorConfig,
    fleet_dir_override: str | None = None,
    *,
    now: datetime.datetime | None = None,
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

    Two things the probe used to *guess* are now read from the file (issue #606):

    * **The driving interval.** The staleness bound was
      ``config.supervisor.full_pass_interval_seconds * 3``, resolved through the
      probe's own config load — a different call from the one the daemon used. The
      pass now records the interval it was driven at, and the bound is computed
      from that recorded value (falling back to config only for a file written
      before the interval was recorded).
    * **The skip reason.** A pass that declines to act (no runners under
      ``managed_root``, an unresolvable root) now records *why*. The probe reports
      the recorded reason instead of asserting "not running unattended (#590)" for
      every stale-or-absent file — "the daemon never reached allocation" and "the
      daemon ran it and found no runners" are different problems with different
      fixes.

    ``now`` is the injectable clock used for the age computation below (issue
    #828): defaults to ``datetime.datetime.now(datetime.timezone.utc)`` when
    not supplied, so production behavior is byte-identical. ``run_doctor``
    samples one ``now`` per doctor run and passes it here and to
    ``_probe_api_worker``.
    """
    allocation = getattr(config, "runner_allocation", None)
    if allocation is None or not allocation.enabled:
        return

    state_dir = fleet_dir(override=fleet_dir_override)
    budget = allocation.max_running_runners

    stamp = load_allocation_stamp(state_dir)
    # Measure staleness against the interval the pass was *actually* driven at,
    # not the one this probe re-resolves — a per-repo layer that sets the
    # interval would otherwise make the probe measure against a cadence the
    # daemon is not running at (issue #606). Fall back to config only for a file
    # written before the interval was recorded.
    recorded_interval = stamp.full_pass_interval_seconds if stamp is not None else None
    interval = max(recorded_interval or config.supervisor.full_pass_interval_seconds, 1)
    # Three intervals: one missed pass is normal jitter (a pass can run long), a
    # sustained gap is not.
    stale_after = interval * 3

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
    resolved_now = now if now is not None else datetime.datetime.now(datetime.timezone.utc)
    age = max(0, int((resolved_now - stamp.updated_at).total_seconds()))
    writer = _allocation_writer_label(stamp.source)

    # A recorded skip reason is the pass saying "I ran, and here is why I did not
    # act." Report that instead of guessing #590: a fresh unattended skip is the
    # daemon reaching allocation and declining — a named, different problem — not
    # the "never reached it" shape #590 describes. A *stale* skip is both: the
    # daemon last found <reason> and has not been back, so the #590 reading joins
    # the recorded reason rather than replacing it.
    #
    # A skip written by a non-unattended source (a manual `charlie runners
    # allocate`, or a file predating provenance) still cannot confirm the daemon
    # is rebalancing — the writer overwrites the same host-wide file, so its
    # skip reason is not evidence the daemon reached allocation. That clause
    # joins the recorded reason the same way staleness does, so a fresh manual
    # skip names *why* it declined *and* flags that it cannot speak for the
    # daemon (issue #590). When both apply (a stale manual skip) the clauses are
    # joined so #590 is cited once rather than duplicated.
    if stamp.skip_reason is not None:
        detail = (
            f"enabled (budget {budget}) but the last pass ({writer}) {age}s ago "
            f"declined to act: {stamp.skip_reason}"
        )
        clauses: list[str] = []
        if stamp.source != UNATTENDED_ALLOCATION_SOURCE:
            clauses.append(
                "this overwrites the same file the unattended pass uses, "
                "so it cannot confirm the daemon is rebalancing"
            )
        if age > stale_after:
            clauses.append(
                f"over the {stale_after}s staleness bound, allocation is not running unattended"
            )
        if clauses:
            detail += " — " + "; ".join(clauses) + " (issue #590)"
        add("runner allocation", False, detail, severity="warning")
        return

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


def _check_provenance_refusal_streak(add: Any, fleet_dir_override: str | None = None) -> None:
    """Surface ci_fleet's provenance refusal streak (issue #1753).

    ``ci_fleet.runner_allocation_pass`` calls ``check_provenance`` before
    actuating and records every non-``ok`` verdict in
    ``provenance-refusals.json`` beside ``runner-allocation.json`` in the
    host-wide fleet dir. ``mismatch`` blocks actuation and escalates after
    three consecutive passes; ``no_anchor`` is an abstention that proceeds but
    escalates once its ``first_seen``→``last_seen`` span exceeds 24h. The
    escalation itself is only ever emitted as a ``runner_allocation_refused``
    event *inside the installed ci_fleet package* — structurally outside
    ``tests/test_event_kind_consumers.py``'s ``src/charlie_work`` scan (issue
    #1364) — so until this check the streak had no consumer at all: a
    ``no_anchor`` streak sat escalated for 22 days while ``charlie doctor``
    reported the fleet green.

    The read goes through ci_fleet's own ``load_refusal_streak`` and the
    verdict through ``RefusalStreak.escalated``, so the escalation thresholds
    (count for ``mismatch``, wall-clock for ``no_anchor``) are never re-derived
    here — a stale copy of the constants would silently disagree with the
    writer the moment ci_fleet retuned them. A corrupt file reads as "no
    streak", matching the loader's deliberate fail-quiet contract: a guard
    whose bookkeeping can manufacture a finding teaches operators to distrust
    its findings. When the file exists but does not parse, the detail says so
    rather than reporting a clean absence.

    Unconditional — deliberately *not* gated on
    ``config.runner_allocation.enabled`` the way ``_check_runner_allocation``
    is. That gate exists because "expected to run" is a per-repo config
    property; the streak file is host-wide state recorded by whichever
    supervisor ran the pass, so gating its consumption on one repo's config
    would re-hide the exact signal this check exists to surface.

    Read-only: ``load_refusal_streak`` parses the file and nothing else.
    """
    state_dir = fleet_dir(override=fleet_dir_override)
    path = state_dir / REFUSAL_STATE_FILENAME
    streak = load_refusal_streak(state_dir)

    if streak is None:
        add(
            "ci_fleet provenance",
            True,
            f"{path} exists but did not parse — read as no recorded streak"
            if path.exists()
            else f"no recorded refusal streak ({path} absent)",
        )
        return

    if streak.escalated:
        add(
            "ci_fleet provenance",
            False,
            f"refusal streak escalated: status={streak.status}, "
            f"consecutive={streak.consecutive}, first_seen={streak.first_seen}, "
            f"last_seen={streak.last_seen} — {streak.detail} (issue #1753)",
            severity="warning",
        )
        return

    add(
        "ci_fleet provenance",
        True,
        f"streak below the escalation threshold: status={streak.status}, "
        f"consecutive={streak.consecutive}, last_seen={streak.last_seen}",
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
    fleet_json_path = layout.fleet_registry_path(override=fleet_dir_override)
    if not fleet_json_path.exists():
        return
    registry = _load_registry(fleet_json_path)
    repos = registry.get("repos", {})
    if not repos:
        return

    fleet_lock_path = layout.fleet_supervisor_lock_path(override=fleet_dir_override)
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
        repo_lock_path = layout.supervisor_lock_path(state_dir)
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
                ["pr", "list", "--state", "all", "--limit", "1", "--json", PROBE_NUMBER_FIELDS],
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
                ["issue", "list", "--state", "all", "--limit", "1", "--json", PROBE_NUMBER_FIELDS],
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


def _check_state_dir_split_brain(add: Any, repo_root: Path, paths: RuntimePaths) -> None:
    """Warn when the default state tree still holds residue after a
    ``runtime.state_dir`` override (issue #712).

    Dispatch creates worktrees under ``layout.default_state_root(repo_root)``
    regardless of ``runtime.state_dir`` (until A2 lands), while
    ``charlie worktree-clean`` and every other subsystem use the *configured*
    ``paths.root`` (see ``layout.py``'s module docstring). When a repo
    overrides ``state_dir``, those are two different directories: cleanup
    enumerates an empty tree while dispatch keeps writing to the default one,
    and nothing ever looks at the default tree again. That is exactly how
    a sibling repo accumulated 74 uncollected worktrees under a 0-byte
    ``events.db`` — a state nobody noticed because nothing diagnosed it.

    This check does not fix the divergence (that is A2's job) — it turns the
    silent residue into a named, warned-about condition. Silent when
    ``state_dir`` is not overridden: the two trees are the same tree, so there
    is nothing to diagnose.
    """
    default_root = layout.default_state_root(repo_root)
    if paths.root == default_root:
        return

    residue: list[str] = []

    default_state_file = layout.state_file_path(default_root)
    if default_state_file.exists():
        residue.append(layout.STATE_FILENAME)

    default_worktrees = layout.worktrees_dir(default_root)
    if default_worktrees.is_dir():
        worktree_count = sum(1 for _ in default_worktrees.iterdir())
        if worktree_count:
            residue.append(f"{layout.WORKTREES_DIRNAME}/ ({worktree_count} entries)")

    default_events_db = _db_path(default_state_file)
    if default_events_db.exists() and default_events_db.stat().st_size > 0:
        residue.append("events.db")

    if not residue:
        return

    add(
        "state dir split-brain",
        False,
        f"configured state_dir is {paths.root}, but the default tree "
        f"{default_root} still contains {', '.join(residue)} — some subsystem "
        f"is writing there while the rest of the orchestrator uses the "
        f"configured tree (issue #712); nothing currently cleans up "
        f"{default_root}, so this residue only grows",
        severity="warning",
    )


def _check_worktrees_root_agreement(
    add: Any, repo_root: Path, paths: RuntimePaths, config: OrchestratorConfig
) -> None:
    """Report the effective worktrees root that dispatch and ``worktree-clean``
    both use.

    Before A2, this was a real pass/fail comparison: dispatch fell back to
    ``layout.worktrees_dir(layout.default_state_root(repo_root))`` (ignoring
    ``runtime.state_dir``) while ``worktree-clean`` swept
    ``layout.worktrees_dir(paths.root)`` (honouring it) — the #712 divergence.
    A2 fixed the root cause by routing both call sites
    (``OrchestratorApp._layout.worktrees`` and
    ``run_worktree_clean_command``'s call into ``clean_worktrees``) through
    the identical ``paths.resolved_layout(config, repo_root).worktrees``. With
    one function computing both sides, there is no second, independently
    derived value left to compare — reintroducing a comparison here would
    necessarily be ``x == x`` (an ``ok=False`` branch that can never fire) and
    would only ever measure whether *this check* also calls the same
    function, not whether dispatch and clean agree. So this is now an
    unconditional report line rather than a pass/fail check; the real
    regression gate for #712 / the ``claude_code.worktrees_dir`` axis lives in
    ``tests/test_paths.py``, exercised against the two actual production call
    sites (``OrchestratorApp._adapter_settings()`` and
    ``run_worktree_clean_command``'s ``clean_worktrees`` call).
    """
    effective_root = resolved_layout(config, repo_root).worktrees
    add(
        "worktrees root",
        True,
        f"{effective_root} (dispatch and `worktree-clean` both resolve here)",
    )


_LANE_FAILURE_LOOKBACK_HOURS = 24


def _check_recent_lane_failures(add: Any, paths: RuntimePaths) -> None:
    """Warn when this repo's own fleet lane recently failed to start (#6-G).

    A lane that fails before ``app.loop()`` runs (e.g. ``load_layered_config``
    raising ``ConfigError`` on an unknown config key, cw 2026-07-29) is caught
    by ``fleet_loop``'s per-repo ``except Exception`` isolation boundary and
    durably recorded here via ``log_event(..., "fleet_pass_config_error",
    ...)`` — see ``fleet_dispatch._record_lane_failure_event``. Without this
    check, that record is only reachable by directly querying events.db; this
    surfaces it in the same preflight report an operator already runs.

    Severity is a warning, not a hard error: the failure already happened and
    may since be fixed — this reports "did this recently happen", not a live
    health gate. (Recording and reading agree only when the fleet registry's
    ``state_dir`` for this repo matches ``paths.state_file`` — see the
    divergence note on ``fleet_dispatch._lane_failure_state_path``.)
    """
    cutoff = (
        (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(hours=_LANE_FAILURE_LOOKBACK_HOURS)
        )
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    events = query_events(paths.state_file, kind="fleet_pass_config_error", since=cutoff, limit=5)
    if not events:
        return
    latest = events[-1]
    payload = latest.get("payload") if isinstance(latest.get("payload"), dict) else {}
    add(
        "recent lane failures",
        False,
        f"{len(events)} fleet-pass lane failure(s) in the last "
        f"{_LANE_FAILURE_LOOKBACK_HOURS}h, most recent at {latest.get('ts')}: "
        f"{payload.get('error')}",
        severity="warning",
    )


_GIT_NETWORK_RETRY_LOOKBACK_HOURS = 24


def _check_git_network_retries(add: Any, paths: RuntimePaths) -> None:
    """Surface recent ``git_network_retry`` events (issue #1777).

    ``git_network_retry`` is registered in ``instrumentation._LEVEL_BY_KIND``
    and emitted from three call sites (``main_ci_reclaim``'s fetch,
    ``self_deploy``'s own pull and its post-repair retry, and the ci-fleet
    sibling pull), but before this check had no consumer beyond an ad-hoc SQL
    query against events.db -- this repo's own "signal without a consumer"
    antipattern, memory-documented as its #1 real defect class. An ``ok:
    false`` row (retries exhausted -- a blip that outlasted the whole backoff
    budget) is exactly the thing that should surface at preflight, and before
    this it surfaced nowhere an operator would see it.

    Queried from two places, because the events live in two different
    events.db files: ``main_ci_reclaim`` logs against *this* repo's own
    ``paths.state_file`` (scoped per-repo like every other fleet-lane event),
    while ``self_deploy`` always operates on the orchestrator's own checkout
    (``orchestrator_root()``) and logs there regardless of which repo
    ``doctor`` was invoked for -- so a doctor run against, say, job-cannon
    still needs to look at charlie-work's own state file to see
    self-deploy's retries. Deduplicated when the two coincide (running
    doctor against the orchestrator checkout itself).

    Read-only and silent when there is nothing to report: ``query_events()``
    itself never raises and returns ``[]`` on any query failure, and an empty
    window adds no check at all (mirrors ``_check_recent_lane_failures``) --
    "no recent retries" is not itself noteworthy.
    """
    cutoff = (
        (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(hours=_GIT_NETWORK_RETRY_LOOKBACK_HOURS)
        )
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state_paths = {paths.state_file, _self_deploy_state_path(orchestrator_root())}
    events: list[dict[str, Any]] = []
    for state_path in state_paths:
        events.extend(query_events(state_path, kind="git_network_retry", since=cutoff))
    if not events:
        return

    def _is_exhausted(event: dict[str, Any]) -> bool:
        payload = event.get("payload")
        return isinstance(payload, dict) and payload.get("ok") is False

    exhausted = [event for event in events if _is_exhausted(event)]
    if exhausted:
        latest = exhausted[-1]
        payload = latest.get("payload") if isinstance(latest.get("payload"), dict) else {}
        add(
            "git network retries",
            False,
            f"{len(exhausted)} of {len(events)} git_network_retry event(s) in the last "
            f"{_GIT_NETWORK_RETRY_LOOKBACK_HOURS}h exhausted retries (blip outlasted the "
            f"backoff budget), most recent site={payload.get('site')} "
            f"attempts={payload.get('attempts')} at {latest.get('ts')}",
            severity="warning",
        )
    else:
        add(
            "git network retries",
            True,
            f"{len(events)} git_network_retry event(s) in the last "
            f"{_GIT_NETWORK_RETRY_LOOKBACK_HOURS}h, all recovered",
        )


_CROSS_REPO_ESCALATION_LOOKBACK_HOURS = 24


def _check_cross_repo_escalations(add: Any, paths: RuntimePaths) -> None:
    """Surface recent ``dispatch_cross_repo_escalated`` events, grouped by
    the sibling repo each escalation pointed at (issue #1789).

    ``CrossRepoGateResult.found_in_repo`` — the managed fleet repo a missing
    candidate was positively matched under — is recorded on the event
    payload at emission (``orchestration/dispatch_state.py``) but before
    this check had no consumer beyond ad-hoc events.db queries: this repo's
    own "signal without a consumer" antipattern, the
    same gap ``_check_git_network_retries`` documents. An operator had to
    open the raw event payload to learn which sibling repo an escalation
    pointed at; aggregating here turns it into the at-a-glance "N
    escalations pointing at repo X" fleet signal — the shape that spots a
    mis-registered or unexpectedly-overlapping repo pair.

    Escalations with no ``found_in_repo`` — a ``cross_repo_scope``
    title-prefix escalation, a confirmed foreign absolute path outside the
    fleet registry, or a legacy payload from before the field existed — are
    counted as unattributed, bucketed by the ``reason`` prefix
    (``cross_repo_scope`` / ``cross_repo_target``) so a scope-gate burst is
    still distinguishable from a foreign-checkout one.

    Warning, not error, and silent when the window is empty: the escalation
    already happened and the parked issue is its own ``human-needed``
    marker — this reports "did this recently happen, and where did it
    point", mirroring ``_check_recent_lane_failures``. Read-only:
    ``query_events()`` never raises and returns ``[]`` on any failure.
    """
    cutoff = (
        (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(hours=_CROSS_REPO_ESCALATION_LOOKBACK_HOURS)
        )
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    events = query_events(paths.state_file, kind="dispatch_cross_repo_escalated", since=cutoff)
    if not events:
        return

    repo_counts: dict[str, int] = {}
    unattributed: dict[str, int] = {}
    for event in events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        found_in_repo = payload.get("found_in_repo")
        if isinstance(found_in_repo, str) and found_in_repo:
            repo_counts[found_in_repo] = repo_counts.get(found_in_repo, 0) + 1
        else:
            reason = payload.get("reason")
            prefix = reason.split(":", 1)[0] if isinstance(reason, str) and reason else "unknown"
            unattributed[prefix] = unattributed.get(prefix, 0) + 1

    detail = (
        f"{len(events)} dispatch_cross_repo_escalated event(s) in the last "
        f"{_CROSS_REPO_ESCALATION_LOOKBACK_HOURS}h, most recent at {events[-1].get('ts')}"
    )
    if repo_counts:
        detail += "; pointing at: " + ", ".join(
            f"{repo} ({count})" for repo, count in sorted(repo_counts.items())
        )
    if unattributed:
        detail += "; unattributed: " + ", ".join(
            f"{prefix} ({count})" for prefix, count in sorted(unattributed.items())
        )
    add(
        "cross-repo escalations",
        False,
        detail,
        severity="warning",
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
    now: datetime.datetime | None = None,
) -> tuple[bool, list[DoctorCheck]]:
    checks: list[DoctorCheck] = []

    # Sampled once for this entire doctor run (issue #828) and threaded into
    # every sub-check that reads the wall clock, instead of each one
    # independently racing it -- see _probe_api_worker's today-derivation and
    # _check_runner_allocation's age computation. Defaults to
    # datetime.datetime.now(datetime.timezone.utc), so production behavior is
    # byte-identical when now is not passed.
    resolved_now = now if now is not None else datetime.datetime.now(datetime.timezone.utc)

    def add(name: str, ok: bool, detail: str, *, severity: str = "error") -> None:
        checks.append(DoctorCheck(name=name, ok=ok, detail=detail, severity=severity))

    # -- backend capability (issue #1706) -------------------------------------
    # The codebase's chosen probe for "this backend cannot host a PR": a
    # local-file backend (``LocalFileGitHub``) has no GitHub remote, so there
    # is no ``gh`` to authenticate, no label registry to bootstrap, no PR
    # surface for the merge-oriented checks, and nothing for --live field
    # probes to interrogate. Discriminate on the capability, not on
    # ``config.local_issues.enabled`` or the ``local/`` name prefix, so every
    # existing test double keeps the gh-shaped path unchanged.
    publishes_prs = publishes_pull_requests(gh)

    # -- environment ---------------------------------------------------------
    if publishes_prs:
        gh_path = shutil.which("gh")
        add(
            "gh on PATH",
            gh_path is not None,
            gh_path or "GitHub CLI `gh` not found on PATH",
        )
        if gh_path:
            try:
                gh.run(["auth", "status"])
                add("gh auth", True, "authenticated")
            except GitHubError as exc:
                add("gh auth", False, str(exc))
    else:
        add(
            "gh on PATH",
            True,
            "not applicable — this backend has no GitHub remote and never runs `gh`",
            severity="warning",
        )
        add(
            "gh auth",
            True,
            "not applicable — no `gh` to authenticate",
            severity="warning",
        )

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

    if publishes_prs:
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
    else:
        add(
            "required checks configured",
            True,
            "not applicable — no pull requests means no merge gate"
            + (
                "; auto_merge settings are inert on this backend"
                if (config.auto_merge.enabled or config.auto_merge.required_checks)
                else ""
            ),
            severity="warning",
        )

    # -- required checks vs live workflow files ------------------------------
    job_names = workflow_job_names(repo_root)
    # Per-job matrix flags (issue #1508): a tolerance match is justified only
    # when the SPECIFIC job it expands from has strategy.matrix, not when any
    # unrelated workflow in the repo has one. The repo-wide boolean this
    # replaced could pass a stale suffixed required-check entry against a
    # non-matrix job merely because a sibling workflow gained a matrix job.
    job_matrix_flags = workflow_job_matrix_flags(repo_root)
    if publishes_prs and config.auto_merge.required_checks:
        if job_names:
            for required in config.auto_merge.required_checks:
                kind = _check_name_match_kind(required, job_names)
                if kind is None:
                    add(
                        f"check name: {required}",
                        False,
                        f"no job in .github/workflows/ reports this name; found: {sorted(job_names)}",
                    )
                elif kind == "exact":
                    add(f"check name: {required}", True, "matches a workflow job")
                else:  # tolerance -- matrix-suffix or reusable-workflow delimiter
                    # Scope the matrix justification to the job(s) that produced
                    # the tolerance match, not the whole repo.
                    bases = _tolerance_match_base_names(required, job_names)
                    matched_has_matrix = any(job_matrix_flags.get(base, False) for base in bases)
                    if matched_has_matrix:
                        add(
                            f"check name: {required}",
                            True,
                            "matches a workflow job via matrix-suffix tolerance "
                            "(matched job has strategy.matrix)",
                        )
                    else:
                        # The merge gate (checks.py) uses EXACT match only, so a
                        # suffixed required-check entry that matches only via
                        # tolerance would be reported `missing` forever -- the
                        # same stale-required-check trap #1507 corrected in the
                        # README. Fail rather than warn: doctor's job is to catch
                        # this before a dispatch wave strands an approved PR
                        # (issue #1508). The matched job has no strategy.matrix,
                        # so the suffix is a stale entry, not a real expansion --
                        # even if some OTHER workflow in the repo has a matrix.
                        add(
                            f"check name: {required}",
                            False,
                            "matches a workflow job only via matrix-suffix tolerance, "
                            "but the matched job has no strategy.matrix -- "
                            "the exact-match merge gate (checks.py) would report this "
                            "required check as missing forever",
                        )
        else:
            add(
                "workflow files",
                False,
                "required_checks configured but no parseable .github/workflows/*.yml found",
                severity="warning",
            )

    # -- labels --------------------------------------------------------------
    if publishes_prs:
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
    else:
        # ``label_list`` on a file-backed backend reports only labels already
        # present in issue frontmatter, and ``label_create`` is a documented
        # no-op — every unused LabelConfig name would read as "missing" with
        # no satisfiable remediation (issue #1706).
        add(
            "github labels",
            True,
            "not applicable — file-backed issues carry free-form frontmatter "
            "labels; there is no label registry to bootstrap",
            severity="warning",
        )

    # -- local-file backend (issue #1706) ------------------------------------
    if not publishes_prs:
        _check_local_issue_backend(add, gh, repo_root, paths, config)

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
    if config.worker.harness == "command" and not config.devin.dispatch_command:
        add("dispatch adapter", False, "adapter is `command` but dispatch_command is empty")
    else:
        add("dispatch adapter", True, config.worker.harness)

    # Report the effective devin-shell worker model so the operator can see
    # what dispatch will actually launch.
    if config.worker.harness == "devin-shell":
        if config.worker.model:
            add(
                "devin-shell worker model",
                True,
                f"config-driven: {config.worker.model}",
            )
        else:
            add(
                "devin-shell worker model",
                True,
                "CLI default (no worker.model configured)",
                severity="warning",
            )

    # claude-code worktrees junction a shared venv in; surface a missing
    # venv_source at preflight rather than deferring it to the first dispatch.
    if config.worker.harness == "claude-code" and config.claude_code.venv_source:
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

    # -- role config summary ------------------------------------------------
    # Always informational -- this section never blocks doctor from passing,
    # it exists purely so an operator can see the resolved worker/reviewer
    # roles in one place, without cross-referencing devin/claude_code/
    # review_dispatch/rescue by hand.
    cross_family_status = "yes" if config.worker.model != config.reviewer.model else "no"
    add(
        "role config",
        True,
        f"worker: harness={config.worker.harness} model={config.worker.model or '(CLI default)'} | "
        f"reviewer: harness={config.reviewer.harness} model={config.reviewer.model or '(CLI default)'} | "
        f"cross-family: {cross_family_status}",
    )

    # -- worker GitHub token (issue #873 Part 2) -----------------------------
    # Config-only, no I/O: reads devin.worker_env/claude_code.worker_env, never
    # the process environment or sanitize_env's output.
    if publishes_prs:
        _check_worker_github_token(add, config)
    else:
        # A worker on a no-remote backend is told never to run `gh`
        # (prompts/worker_sections/local_no_merge_contract.md), so the missing
        # scoped-token finding is a false positive here (issue #1706).
        add(
            "worker GitHub token",
            True,
            "not applicable — workers on this backend never run `gh`, so "
            "sanitize_env has no token to restore",
            severity="warning",
        )

    # -- api worker observability (issue #483) ------------------------------
    # Always runs (not gated on --adapter-probe): these are config/environment
    # checks, not external CLI probes. Read-only — never mutates the ledger.
    _probe_api_worker(add, paths, config, now=resolved_now)

    if adapter_probe:
        _probe_adapter(add, repo_root, config)
        _surface_sessions(add, repo_root, config)
        # Issue #1346: surface the watchdog's corroboration probe for
        # in-progress workers so alive-but-polling (stale sidecar log, fresh
        # sessions.db / per-PID log) is visibly distinguishable from genuinely
        # stalled. Read-only, same code path as the watchdog.
        _surface_in_progress_corroboration(add, repo_root, config, now=resolved_now)

    if live:
        if publishes_prs:
            _validate_gh_field_lists(add, gh)
        else:
            add(
                "gh field lists",
                True,
                "skipped — this backend has no `gh` to probe",
                severity="warning",
            )

    # -- review-to-verdict path (gap that let 34 reviews sit unread) ---------
    # review_dispatch.enabled and rescue.enabled are the two paths that call
    # record_review() automatically (the rescue tier's own reviewer launch
    # calls it directly from _process_rescue_review, workflow.py). If both
    # are off there is no automated route from a completed review to a
    # recorded verdict at all: PRs pile up in "reviewing" with valid reports
    # and no decision, invisible to any other check here.
    if publishes_prs:
        has_review_to_verdict_path = config.review_dispatch.enabled or config.rescue.enabled
        add(
            "review-to-verdict path",
            has_review_to_verdict_path,
            "ok"
            if has_review_to_verdict_path
            else (
                "review_dispatch.enabled and rescue.enabled are both "
                "false: no automated path from review to verdict; PRs accumulate "
                "in reviewing with valid reports and no decision"
            ),
        )
    else:
        # The check's premise is PR-shaped: without pull requests there is no
        # "reviewing" lane to accumulate in — parked branches are handed to a
        # human via the review-ready label instead (issue #1706).
        add(
            "review-to-verdict path",
            True,
            "not applicable — no pull requests; completed work is parked on its "
            "branch under the review-ready label for a human to merge",
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

    # -- state-dir split-brain (issue #712) ----------------------------------
    # Read-only: only stats/lists the default tree, never touches it.
    _check_state_dir_split_brain(add, repo_root, paths)
    _check_worktrees_root_agreement(add, repo_root, paths, config)

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
    _check_runner_allocation(add, config, fleet_dir_override=fleet_dir_override, now=resolved_now)

    # -- ci_fleet provenance refusal streak (issue #1753) ---------------------
    # Read-only: parses the host-wide streak file via ci_fleet's own loader.
    # Not gated on runner_allocation.enabled — the file is host-wide state
    # written by whichever supervisor ran the pass, and gating its consumer on
    # one repo's config would re-hide the signal this check exists to surface.
    _check_provenance_refusal_streak(add, fleet_dir_override=fleet_dir_override)

    # -- recent lane-startup failures (#6-G) ---------------------------------
    # Read-only: queries this repo's own events.db for past
    # fleet_pass_config_error records. Never raises: query_events() itself
    # returns [] rather than propagating on a query error.
    _check_recent_lane_failures(add, paths)

    # -- recent git_network_retry events (issue #1777) -----------------------
    # Read-only: queries events.db for both this repo's own state file and
    # the orchestrator's self-deploy state file. Flags exhausted retries
    # (ok: false) as a warning; adds nothing when the window is empty.
    _check_git_network_retries(add, paths)

    # -- recent dispatch_cross_repo_escalated events (issue #1789) -----------
    # Read-only: aggregates this repo's own events.db by the escalation
    # payload's found_in_repo field -- the sibling repo a missing candidate
    # was positively matched under. Warning-severity, silent when the
    # window is empty.
    _check_cross_repo_escalations(add, paths)

    hard_failures = [check for check in checks if not check.ok and check.severity == "error"]
    return (not hard_failures, checks)
