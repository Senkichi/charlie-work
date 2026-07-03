"""Preflight diagnostics: verify the environment, config, labels, and CI-check
names before a dispatch wave, instead of discovering mismatches mid-run.

The required-check verification derives job names from the consumer repo's
``.github/workflows/*.yml`` at run time — the check list itself stays in
config, but its validity is never asserted by hand.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import OrchestratorConfig
from .github import (
    GitHub,
    GitHubError,
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
    add("launched sessions", not failed, detail, severity="warning")


def _validate_gh_field_lists(add: Any, gh: GitHub) -> None:
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
    gh: GitHub,
    *,
    adapter_probe: bool = False,
    live: bool = False,
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

    hard_failures = [check for check in checks if not check.ok and check.severity == "error"]
    return (not hard_failures, checks)
