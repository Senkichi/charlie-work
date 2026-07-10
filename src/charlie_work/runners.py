"""Runner pool observability for self-hosted GitHub Actions runners.

This module provides read-only observability for runner pools:
- GitHub Actions runner status (online/busy/offline)
- CI workflow queue depth
- Host resource metrics (CPU, RAM)
- Derived pressure classification (saturated/balanced/idle)

Scaling actions are deferred to future issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .config import RunnerScalingConfig
from .github import GitHub


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


def observe_runner_pool(
    gh: GitHub,
    config: RunnerScalingConfig,
    *,
    workflow_filename: str | None = None,
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
            If None, uses a default heuristic (main branch CI).

    Returns:
        RunnerPoolState snapshot

    Raises:
        GitHubError: If GitHub API calls fail
    """
    from datetime import datetime, timezone

    # Query GitHub Actions runners
    runners_data = gh.run(["api", "repos/{owner}/{repo}/actions/runners"], json_output=True)
    runners = runners_data.get("runners", []) if runners_data else []

    # Classify runners
    total_runners = len(runners)
    online_runners = sum(1 for r in runners if r.get("status") == "online")
    busy_runners = sum(1 for r in runners if r.get("busy") is True)
    idle_runners = online_runners - busy_runners

    # Query workflow run queue depth
    # Default to main branch CI if no specific workflow provided
    if workflow_filename is None:
        # Use a simple heuristic: count queued and in_progress runs for main branch
        runs_data = gh.run(
            ["api", "repos/{owner}/{repo}/actions/runs", "--branch=main", "--per-page=100"],
            json_output=True,
        )
        runs = runs_data.get("workflow_runs", []) if runs_data else []
        queued_jobs = sum(1 for r in runs if r.get("status") in ("queued", "pending"))
        in_progress_jobs = sum(1 for r in runs if r.get("status") == "in_progress")
    else:
        # Filter by specific workflow filename
        runs_data = gh.run(
            [
                "api",
                "repos/{owner}/{repo}/actions/runs",
                f"--workflow={workflow_filename}",
                "--per-page=100",
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
        # Fallback if psutil unavailable or metrics fail
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
