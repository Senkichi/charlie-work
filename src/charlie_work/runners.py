"""Runner pool observability and provisioning for self-hosted GitHub Actions runners.

This module provides:
- Read-only observability for runner pools (status, queue depth, host metrics)
- Scale-up provisioning engine (token mint, package extract, unattended config, decontaminated launch)
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .config import RunnerScalingConfig
from .github import GitHub
from .subprocess_runner import RunResult, run_captured


class PoolPressure(str, Enum):
    """Runner pool pressure classification."""

    SATURATED = "saturated"  # No headroom: queue depth > 0 or host resources constrained
    BALANCED = "balanced"  # Healthy: runners available, resources adequate
    IDLE = "idle"  # Underutilized: no queue, excess capacity


@dataclass(frozen=True)
class RunnerPoolState:
    """Snapshot of runner pool state.

    Read-only observability data structure. All fields are derived from
    GitHub API and host metrics; no scaling actions are performed.
    """

    # Pool size metrics
    total_runners: int  # Total registered runners
    online_runners: int  # Runners with status 'online'
    busy_runners: int  # Runners currently executing jobs
    idle_runners: int  # Online runners not executing jobs

    # Queue depth
    queued_jobs: int  # Jobs waiting for a runner
    in_progress_jobs: int  # Jobs currently running

    # Host resource headroom
    free_ram_gb: float  # Free RAM in GB
    cpu_percent: float  # CPU utilization percentage

    # Derived classification
    pressure: PoolPressure  # Overall pool pressure classification

    # Timestamp
    timestamp: str  # ISO 8601 timestamp of the snapshot


# Marker file name that identifies charlie-managed runner directories
CHARLIE_MANAGED_MARKER = ".charlie-managed"


@dataclass(frozen=True)
class ProvisioningResult:
    """Result of a runner provisioning attempt.

    Follows the RunResult pattern: errors as values, never exceptions.
    """

    ok: bool
    runner_name: str | None = None
    runner_dir: Path | None = None
    error: str | None = None
    # Dry-run mode: planned actions without execution
    dry_run: bool = False
    dry_run_actions: list[str] = field(default_factory=list)


def observe_runner_pool(
    gh: GitHub,
    config: RunnerScalingConfig,
    *,
    workflow_filename: str | None = None,
    default_branch: str | None = None,
) -> RunnerPoolState:
    """Observe runner pool state for a repository.

    This is a read-only function that collects observability data:
    - Queries GitHub Actions runners via gh API
    - Queries CI workflow run queue depth
    - Samples host CPU and RAM metrics via psutil
    - Derives pressure classification

    Args:
        gh: GitHub client instance
        config: Runner scaling configuration
        workflow_filename: Optional CI workflow filename to filter queue depth.
            If None, counts runs on the repository's default branch instead.
        default_branch: Optional default branch name for the queue-depth query.
            Only consulted when ``workflow_filename`` is None. If also None, the
            branch is resolved once via the repository metadata endpoint.

    Returns:
        RunnerPoolState snapshot

    Raises:
        GitHubError: If GitHub API calls fail
    """
    from datetime import datetime, timezone

    # Query GitHub Actions runners.
    # Endpoint is passed verbatim to `gh api`; the {owner}/{repo} placeholders
    # are substituted by gh from the current repo context.
    runners_data = gh.run(["api", "repos/{owner}/{repo}/actions/runners"], json_output=True)
    runners = runners_data.get("runners", []) if runners_data else []

    # Classify runners
    total_runners = len(runners)
    online_runners = sum(1 for r in runners if r.get("status") == "online")
    busy_runners = sum(1 for r in runners if r.get("busy") is True)
    idle_runners = online_runners - busy_runners

    # Query workflow run queue depth.
    # NOTE: `gh api` takes query params in the endpoint string (e.g. ?per_page=100),
    # NOT as gh-run-list-style flags (--branch/--per-page/--workflow); those flags
    # are silently rejected by `gh api` and would fail at runtime.
    if workflow_filename is None:
        # No specific workflow: count runs on the repository's default branch.
        # Resolve the default branch once if the caller did not supply it.
        if default_branch is None:
            repo_data = gh.run(["api", "repos/{owner}/{repo}"], json_output=True)
            default_branch = repo_data.get("default_branch", "main") if repo_data else "main"
        runs_data = gh.run(
            [
                "api",
                f"repos/{{owner}}/{{repo}}/actions/runs?branch={default_branch}&per_page=100",
            ],
            json_output=True,
        )
        runs = runs_data.get("workflow_runs", []) if runs_data else []
        queued_jobs = sum(1 for r in runs if r.get("status") in ("queued", "pending"))
        in_progress_jobs = sum(1 for r in runs if r.get("status") == "in_progress")
    else:
        # Filter by specific workflow filename. The REST shape scopes runs under
        # the workflow resource; there is no `workflow=` query param on /actions/runs.
        runs_data = gh.run(
            [
                "api",
                f"repos/{{owner}}/{{repo}}/actions/workflows/{workflow_filename}/runs?per_page=100",
            ],
            json_output=True,
        )
        runs = runs_data.get("workflow_runs", []) if runs_data else []
        queued_jobs = sum(1 for r in runs if r.get("status") in ("queued", "pending"))
        in_progress_jobs = sum(1 for r in runs if r.get("status") == "in_progress")

    # Sample host metrics
    try:
        import psutil

        # RAM: free in GB
        ram = psutil.virtual_memory()
        free_ram_gb = ram.available / (1024**3)

        # CPU: utilization percentage
        cpu_percent = psutil.cpu_percent(interval=0.1)
    except (ImportError, OSError):
        # Fallback if psutil unavailable or metrics fail.
        # The 0.0 fallbacks are intentional: free_ram_gb=0.0 (< min_free_ram_gb)
        # forces a SATURATED classification, so unknown host headroom is treated
        # as no-headroom — fail-conservative for future scale-up decisions.
        free_ram_gb = 0.0
        cpu_percent = 0.0

    # Derive pressure classification
    pressure = _classify_pressure(
        queued_jobs=queued_jobs,
        idle_runners=idle_runners,
        free_ram_gb=free_ram_gb,
        cpu_percent=cpu_percent,
        config=config,
    )

    timestamp = datetime.now(timezone.utc).isoformat()

    return RunnerPoolState(
        total_runners=total_runners,
        online_runners=online_runners,
        busy_runners=busy_runners,
        idle_runners=idle_runners,
        queued_jobs=queued_jobs,
        in_progress_jobs=in_progress_jobs,
        free_ram_gb=free_ram_gb,
        cpu_percent=cpu_percent,
        pressure=pressure,
        timestamp=timestamp,
    )


def _classify_pressure(
    queued_jobs: int,
    idle_runners: int,
    free_ram_gb: float,
    cpu_percent: float,
    config: RunnerScalingConfig,
) -> PoolPressure:
    """Classify pool pressure based on queue and resource metrics.

    Args:
        queued_jobs: Number of jobs waiting for a runner
        idle_runners: Number of online runners not executing jobs
        free_ram_gb: Free RAM in GB
        cpu_percent: CPU utilization percentage
        config: Runner scaling configuration with thresholds

    Returns:
        PoolPressure classification
    """
    # Saturated: queue exists OR resources constrained
    if queued_jobs > 0:
        return PoolPressure.SATURATED
    if free_ram_gb < config.min_free_ram_gb:
        return PoolPressure.SATURATED
    if cpu_percent > config.max_host_cpu_pct:
        return PoolPressure.SATURATED

    # Idle: no queue AND excess capacity
    if idle_runners > 0 and queued_jobs == 0:
        return PoolPressure.IDLE

    # Balanced: healthy state
    return PoolPressure.BALANCED


def _mint_registration_token(gh: GitHub) -> dict[str, Any] | None:
    """Mint a GitHub Actions runner registration token.

    The token is returned in the JSON response and must never be logged or persisted.

    Args:
        gh: GitHub client instance

    Returns:
        Dict with token if successful, None if failed
    """
    result = gh.run(
        ["api", "-X", "POST", "repos/{owner}/{repo}/actions/runners/registration-token"],
        json_output=True,
    )
    # The token is in the result under the "token" key
    # We return it as-is; the caller must extract and use it without logging
    return result if isinstance(result, dict) else None


def _allocate_runner_dir(
    managed_root: Path,
    runner_dir_prefix: str,
    *,
    dry_run: bool = False,
) -> tuple[Path, int]:
    """Allocate the next runner directory by scanning for existing instances.

    Derives the next index {n} from what exists, never a stored counter.

    Args:
        managed_root: Root directory for runner instances
        runner_dir_prefix: Directory name prefix (e.g., "jc-")
        dry_run: If True, derive the next index without creating the directory

    Returns:
        Tuple of (runner_dir, next_index)
    """
    managed_root = Path(managed_root)
    if not managed_root.exists():
        if dry_run:
            # In dry-run mode, don't create the managed_root
            pass
        else:
            managed_root.mkdir(parents=True, exist_ok=True)

    # Scan for existing runner directories
    existing_indices = []
    if managed_root.exists():
        for entry in managed_root.iterdir():
            if entry.is_dir() and entry.name.startswith(runner_dir_prefix):
                # Extract index from directory name (e.g., "jc-1" -> 1)
                suffix = entry.name[len(runner_dir_prefix) :]
                try:
                    index = int(suffix)
                    existing_indices.append(index)
                except ValueError:
                    # Not a numbered directory, skip
                    pass

    # Allocate next index
    next_index = max(existing_indices) + 1 if existing_indices else 1
    runner_dir = managed_root / f"{runner_dir_prefix}{next_index}"

    # Create the directory only in real mode
    if not dry_run:
        runner_dir.mkdir(parents=True, exist_ok=True)

    return runner_dir, next_index


def _write_charlie_managed_marker(runner_dir: Path) -> None:
    """Write the charlie-managed marker file to identify ownership.

    The marker is written BEFORE configuration so cleanup paths can always
    identify ownership. Marker-less dirs are never touched by cleanup.

    Args:
        runner_dir: Path to the runner directory
    """
    marker_file = runner_dir / CHARLIE_MANAGED_MARKER
    marker_file.write_text("charlie-work managed runner\n")


def _extract_runner_package(package_zip: Path, runner_dir: Path) -> RunResult:
    """Extract the runner package zip to the runner directory.

    Args:
        package_zip: Path to the runner package zip file
        runner_dir: Path to the runner directory

    Returns:
        RunResult indicating success or failure
    """
    try:
        with zipfile.ZipFile(package_zip, "r") as zip_ref:
            zip_ref.extractall(runner_dir)
        return RunResult(returncode=0, stdout="", stderr="", error=None)
    except Exception as exc:
        return RunResult(
            returncode=None,
            stdout="",
            stderr="",
            error=f"Failed to extract package: {exc}",
        )


def _configure_runner(
    runner_dir: Path,
    url: str,
    token: str,
    name: str,
) -> RunResult:
    """Configure the runner with unattended mode.

    Runs config.cmd with --unattended --url --token --name --work _work

    Args:
        runner_dir: Path to the runner directory
        url: GitHub repository URL
        token: Registration token (never logged)
        name: Runner name
        work: Work directory name (default: _work)

    Returns:
        RunResult from the configuration command
    """
    config_cmd = runner_dir / "config.cmd"
    if not config_cmd.exists():
        return RunResult(
            returncode=None,
            stdout="",
            stderr="",
            error=f"config.cmd not found in {runner_dir}",
        )

    # Build the command (token is passed as argument, never logged)
    cmd = [
        str(config_cmd),
        "--unattended",
        "--url",
        url,
        "--token",
        token,  # Token passed as argument; subprocess_runner handles it securely
        "--name",
        name,
        "--work",
        "_work",
    ]

    result = run_captured(
        command=cmd,
        cwd=runner_dir,
        timeout_seconds=300,
    )

    return result


def _sanitize_env() -> dict[str, str]:
    """Strip environment variables that could contaminate the runner launch.

    Removes UV_*, VIRTUAL_ENV, PYTHON*, PIP_*, CLAUDE* to prevent dev-shell
    environment contamination (2026-07-08 lesson).

    Returns:
        Sanitized environment dictionary
    """
    env = os.environ.copy()
    prefixes_to_strip = [
        "UV_",
        "VIRTUAL_ENV",
        "PYTHON",
        "PIP_",
        "CLAUDE",
    ]
    keys_to_remove = []
    for key in env:
        if any(key.startswith(prefix) for prefix in prefixes_to_strip):
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del env[key]
    return env


def _launch_runner(runner_dir: Path) -> RunResult:
    """Launch the runner listener with a decontaminated environment.

    Strips UV_*, VIRTUAL_ENV, PYTHON*, PIP_*, CLAUDE* from the child env.

    Args:
        runner_dir: Path to the runner directory

    Returns:
        RunResult from the launch command (non-blocking)
    """
    run_cmd = runner_dir / "run.cmd"
    if not run_cmd.exists():
        return RunResult(
            returncode=None,
            stdout="",
            stderr="",
            error=f"run.cmd not found in {runner_dir}",
        )

    # Use subprocess.Popen directly for non-blocking launch
    # The adapter invariant: never call process.wait() or process.communicate()
    import subprocess

    try:
        env = _sanitize_env()
        subprocess.Popen(
            [str(run_cmd)],
            cwd=str(runner_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return RunResult(returncode=0, stdout="", stderr="", error=None)
    except Exception as exc:
        return RunResult(
            returncode=None,
            stdout="",
            stderr="",
            error=f"Failed to launch runner: {exc}",
        )


def _verify_runner_online(
    gh: GitHub,
    runner_name: str,
    max_retries: int = 30,
    retry_interval_seconds: int = 5,
) -> RunResult:
    """Poll the runners API until the new runner reports online.

    Args:
        gh: GitHub client instance
        runner_name: Name of the runner to verify
        max_retries: Maximum number of polling attempts
        retry_interval_seconds: Seconds between retries

    Returns:
        RunResult indicating if the runner came online
    """
    import time

    for attempt in range(max_retries):
        runners_data = gh.run(["api", "repos/{owner}/{repo}/actions/runners"], json_output=True)
        runners = runners_data.get("runners", []) if isinstance(runners_data, dict) else []

        for runner in runners:
            if runner.get("name") == runner_name:
                status = runner.get("status")
                if status == "online":
                    return RunResult(
                        returncode=0,
                        stdout=f"Runner {runner_name} is online",
                        stderr="",
                        error=None,
                    )
                # Runner exists but not online yet, continue polling

        if attempt < max_retries - 1:
            time.sleep(retry_interval_seconds)

    return RunResult(
        returncode=None,
        stdout="",
        stderr="",
        error=f"Runner {runner_name} did not come online after {max_retries} retries",
    )


def _cleanup_runner_dir(runner_dir: Path) -> None:
    """Remove a runner directory if it has the charlie-managed marker.

    Marker-less dirs are never touched (safety invariant).

    Args:
        runner_dir: Path to the runner directory
    """
    marker_file = runner_dir / CHARLIE_MANAGED_MARKER
    if marker_file.exists():
        import shutil

        shutil.rmtree(runner_dir)


def provision_runner(
    gh: GitHub,
    config: RunnerScalingConfig,
    busy_jobs: int,
    *,
    dry_run: bool = False,
) -> ProvisioningResult:
    """Provision a new self-hosted GitHub Actions runner.

    Scale-up provisioning engine, Windows-first:
    1. Mint a registration token via GitHub API
    2. Allocate the next runner dir by scanning managed_root
    3. Extract package_zip, run config.cmd with unattended mode
    4. Launch the listener with decontaminated environment
    5. Verify the runner reports online via polling

    Guardrails:
    - Refuse to provision beyond max_runners
    - Refuse when projected RAM would breach min_free_ram_gb
    - Refuse when the feature is disabled

    Args:
        gh: GitHub client instance
        config: Runner scaling configuration
        busy_jobs: Current number of busy jobs (for RAM projection)
        dry_run: If True, print planned actions without executing

    Returns:
        ProvisioningResult with outcome
    """
    actions: list[str] = []

    # Guardrail: feature disabled
    if not config.enabled:
        return ProvisioningResult(
            ok=False,
            error="Runner scaling is disabled in config",
        )

    # Guardrail: max_runners
    runners_data = gh.run(["api", "repos/{owner}/{repo}/actions/runners"], json_output=True)
    runners = runners_data.get("runners", []) if isinstance(runners_data, dict) else []
    current_runner_count = len(runners)

    if current_runner_count >= config.max_runners:
        return ProvisioningResult(
            ok=False,
            error=f"Max runners ({config.max_runners}) already reached",
        )

    # Guardrail: RAM headroom
    try:
        import psutil

        ram = psutil.virtual_memory()
        free_ram_gb = ram.available / (1024**3)
    except (ImportError, OSError):
        free_ram_gb = 0.0  # Fail-conservative

    projected_ram_usage = (busy_jobs + 1) * config.ram_per_job_gb
    if free_ram_gb - projected_ram_usage < config.min_free_ram_gb:
        return ProvisioningResult(
            ok=False,
            error=f"Insufficient RAM: {free_ram_gb:.2f}GB free, need {config.min_free_ram_gb:.2f}GB after provisioning",
        )

    # Step 1: Mint registration token
    if dry_run:
        actions.append("Mint registration token via GitHub API")
    else:
        token_data = _mint_registration_token(gh)
        if token_data is None:
            return ProvisioningResult(
                ok=False,
                error="Failed to mint token",
            )
        token = token_data.get("token", "")
        if not token:
            return ProvisioningResult(
                ok=False,
                error="Token not found in API response",
            )

    # Step 2: Allocate runner directory
    managed_root = Path(config.managed_root)
    runner_dir, next_index = _allocate_runner_dir(
        managed_root,
        config.runner_dir_prefix,
        dry_run=dry_run,
    )
    runner_name = config.runner_name_template.format(n=next_index)

    if dry_run:
        actions.append(f"Allocate runner directory: {runner_dir}")
        actions.append(f"Runner name: {runner_name}")
    else:
        # Write marker BEFORE configuration (ownership invariant)
        _write_charlie_managed_marker(runner_dir)

    # Step 3: Extract package
    package_zip = Path(config.package_zip)
    if not package_zip.exists():
        if dry_run:
            actions.append(f"Extract package: {package_zip} (file not found)")
            return ProvisioningResult(
                ok=False,
                error=f"Package zip not found: {package_zip}",
                dry_run=True,
                dry_run_actions=actions,
            )
        else:
            # Cleanup: remove the just-created dir with marker
            _cleanup_runner_dir(runner_dir)
            return ProvisioningResult(
                ok=False,
                error=f"Package zip not found: {package_zip}",
            )

    if dry_run:
        actions.append(f"Extract package: {package_zip} -> {runner_dir}")
    else:
        extract_result = _extract_runner_package(package_zip, runner_dir)
        if not extract_result.ok:
            # Cleanup: remove the just-created dir with marker
            _cleanup_runner_dir(runner_dir)
            return ProvisioningResult(
                ok=False,
                error=f"Failed to extract package: {extract_result.error}",
            )

    # Step 4: Configure runner
    # Get repository URL from GitHub API
    repo_data = gh.run(["api", "repos/{owner}/{repo}"], json_output=True)
    repo_url = repo_data.get("html_url", "") if isinstance(repo_data, dict) else ""

    if dry_run:
        actions.append(
            f"Configure runner: config.cmd --unattended --url {repo_url} --token *** --name {runner_name} --work _work"
        )
    else:
        config_result = _configure_runner(
            runner_dir,
            repo_url,
            token,
            runner_name,
        )
        if not config_result.ok:
            # Cleanup: remove the just-created dir with marker
            _cleanup_runner_dir(runner_dir)
            return ProvisioningResult(
                ok=False,
                error=f"Failed to configure runner: {config_result.error}",
            )

    # Step 5: Launch runner
    if dry_run:
        actions.append(f"Launch runner with decontaminated environment: {runner_dir}/run.cmd")
    else:
        launch_result = _launch_runner(runner_dir)
        if not launch_result.ok:
            # Cleanup: remove the just-created dir with marker
            _cleanup_runner_dir(runner_dir)
            return ProvisioningResult(
                ok=False,
                error=f"Failed to launch runner: {launch_result.error}",
            )

    # Step 6: Verify runner online
    if dry_run:
        actions.append(f"Verify runner online via polling: {runner_name}")
        return ProvisioningResult(
            ok=True,
            runner_name=runner_name,
            runner_dir=runner_dir,
            dry_run=True,
            dry_run_actions=actions,
        )
    else:
        verify_result = _verify_runner_online(gh, runner_name)
        if not verify_result.ok:
            # Don't cleanup on verification failure - the runner may still be starting
            # The operator can inspect the state manually
            return ProvisioningResult(
                ok=False,
                error=f"Runner verification failed: {verify_result.error}",
                runner_name=runner_name,
                runner_dir=runner_dir,
            )

        return ProvisioningResult(
            ok=True,
            runner_name=runner_name,
            runner_dir=runner_dir,
        )


def format_runner_pool_state(state: RunnerPoolState) -> dict[str, Any]:
    """Format RunnerPoolState for display (CLI/JSON).

    Args:
        state: Runner pool state snapshot

    Returns:
        Dictionary with formatted state data
    """
    return {
        "pool_size": {
            "total": state.total_runners,
            "online": state.online_runners,
            "busy": state.busy_runners,
            "idle": state.idle_runners,
        },
        "queue_depth": {
            "queued": state.queued_jobs,
            "in_progress": state.in_progress_jobs,
        },
        "host_headroom": {
            "free_ram_gb": round(state.free_ram_gb, 2),
            "cpu_percent": round(state.cpu_percent, 1),
        },
        "pressure": state.pressure.value,
        "timestamp": state.timestamp,
    }
