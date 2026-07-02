"""Preflight diagnostics: verify the environment, config, labels, and CI-check
names before a dispatch wave, instead of discovering mismatches mid-run.

The required-check verification derives job names from the consumer repo's
``.github/workflows/*.yml`` at run time — the check list itself stays in
config, but its validity is never asserted by hand.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import OrchestratorConfig
from .github import GitHub, GitHubError
from .paths import RuntimePaths
from .prompts import resolve_template
from .state import load_state


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


def run_doctor(
    repo_root: Path,
    paths: RuntimePaths,
    config: OrchestratorConfig,
    config_path: Path | None,
    gh: GitHub,
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
    try:
        state = load_state(paths.state_file)
        add(
            "state file",
            True,
            f"{paths.state_file} (issues: {len(state.get('issues', {}))}, "
            f"prs: {len(state.get('prs', {}))})",
        )
    except (OSError, ValueError) as exc:
        add("state file", False, f"{paths.state_file}: {exc}")

    # -- adapters ------------------------------------------------------------
    if config.devin.adapter == "command" and not config.devin.dispatch_command:
        add("dispatch adapter", False, "adapter is `command` but dispatch_command is empty")
    else:
        add("dispatch adapter", True, config.devin.adapter)

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
