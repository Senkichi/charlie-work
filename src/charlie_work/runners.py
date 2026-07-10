"""Runner pool observability for self-hosted GitHub Actions runners.

This module provides read-only observability for runner pools:
- GitHub Actions runner status (online/busy/offline)
- CI workflow queue depth
- Host resource metrics (CPU, RAM)
- Derived pressure classification (saturated/balanced/idle)
- Idle detection via persistent pool samples

Scaling actions are deferred to future issues.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from .config import RunnerScalingConfig
from .github import GitHub, GitHubError


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


@dataclass(frozen=True)
class PoolSample:
    """A single pool state sample for idle detection.

    Used to persist pool state across invoke-per-pass runs to determine
    if the pool has been idle for a continuous period.
    """

    timestamp: str  # ISO 8601 timestamp of the sample
    busy: bool  # Whether any runner was busy at this sample
    queued_jobs: int  # Number of queued jobs at this sample


def save_pool_sample(state_dir: Path, sample: PoolSample) -> None:
    """Append a pool sample to the persistent samples file.

    Uses atomic temp-file + replace pattern for cross-process safety.

    Args:
        state_dir: The state directory (e.g., .var/charlie-work)
        sample: The pool sample to append
    """
    samples_path = state_dir / "runner-pool-samples.jsonl"
    samples_path.parent.mkdir(parents=True, exist_ok=True)

    # Append to file (atomic for single-line appends)
    with samples_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": sample.timestamp, "busy": sample.busy, "queued_jobs": sample.queued_jobs}) + "\n")


def load_pool_samples(state_dir: Path, max_age_minutes: int = 60) -> list[PoolSample]:
    """Load pool samples from the persistent samples file.

    Filters out samples older than max_age_minutes to prevent unbounded growth.

    Args:
        state_dir: The state directory (e.g., .var/charlie-work)
        max_age_minutes: Maximum age of samples to return (default 60 minutes)

    Returns:
        List of pool samples, sorted by timestamp (oldest first)
    """
    samples_path = state_dir / "runner-pool-samples.jsonl"
    if not samples_path.exists():
        return []

    samples: list[PoolSample] = []
    cutoff = datetime.now(UTC) - timedelta(minutes=max_age_minutes)

    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                timestamp = data.get("timestamp", "")
                busy = data.get("busy", False)
                queued_jobs = data.get("queued_jobs", 0)

                # Parse timestamp and check age
                try:
                    sample_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    if sample_time < cutoff:
                        continue  # Skip old samples
                except (ValueError, TypeError):
                    continue  # Skip malformed timestamps

                samples.append(PoolSample(timestamp=timestamp, busy=busy, queued_jobs=queued_jobs))
            except (json.JSONDecodeError, TypeError):
                continue  # Skip malformed lines

    return sorted(samples, key=lambda s: s.timestamp)


def cleanup_pool_samples(state_dir: Path, max_age_minutes: int = 60) -> None:
    """Remove old samples from the persistent samples file.

    Rewrites the file with only recent samples to prevent unbounded growth.

    Args:
        state_dir: The state directory (e.g., .var/charlie-work)
        max_age_minutes: Maximum age of samples to keep (default 60 minutes)
    """
    samples_path = state_dir / "runner-pool-samples.jsonl"
    if not samples_path.exists():
        return

    recent_samples = load_pool_samples(state_dir, max_age_minutes)

    # Atomic rewrite
    tmp_path = samples_path.with_suffix(samples_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for sample in recent_samples:
            f.write(json.dumps({"timestamp": sample.timestamp, "busy": sample.busy, "queued_jobs": sample.queued_jobs}) + "\n")
    tmp_path.replace(samples_path)


def is_pool_idle_for_minutes(state_dir: Path, idle_minutes: int) -> bool:
    """Check if the pool has been idle for a continuous period.

    A pool is considered idle if:
    - No samples show busy=True
    - No samples show queued_jobs > 0
    - Samples span at least idle_minutes

    Args:
        state_dir: The state directory (e.g., .var/charlie-work)
        idle_minutes: Required continuous idle duration in minutes

    Returns:
        True if the pool has been idle for the required duration, False otherwise
    """
    samples = load_pool_samples(state_dir, max_age_minutes=idle_minutes + 5)  # Load slightly more than needed

    if not samples:
        return False  # No samples yet, cannot determine idle state

    # Check if any sample shows activity
    for sample in samples:
        if sample.busy or sample.queued_jobs > 0:
            return False  # Pool was active at some point

    # Check time span
    if len(samples) < 2:
        return False  # Need at least 2 samples to determine duration

    oldest = datetime.fromisoformat(samples[0].timestamp.replace("Z", "+00:00"))
    newest = datetime.fromisoformat(samples[-1].timestamp.replace("Z", "+00:00"))
    duration = (newest - oldest).total_seconds() / 60  # Convert to minutes

    return duration >= idle_minutes


@dataclass(frozen=True)
class RunnerDir:
    """A managed runner directory with metadata."""

    path: Path  # Path to the runner directory
    name: str  # Runner name
    is_managed: bool  # Whether this is a charlie-managed runner (has marker file)


def discover_managed_runners(managed_root: Path, runner_dir_prefix: str) -> list[RunnerDir]:
    """Discover all charlie-managed runner directories under managed_root.

    A runner is considered charlie-managed if it:
    - Has a directory name matching the prefix
    - Contains a .charlie-managed marker file

    Args:
        managed_root: Root directory where runner instances are managed
        runner_dir_prefix: Directory name prefix for runner instances

    Returns:
        List of RunnerDir objects for all managed runners
    """
    if not managed_root.exists():
        return []

    marker_file = ".charlie-managed"
    managed_runners: list[RunnerDir] = []

    for entry in managed_root.iterdir():
        if not entry.is_dir():
            continue
        if not entry.name.startswith(runner_dir_prefix):
            continue

        # Check for charlie-managed marker
        marker_path = entry / marker_file
        is_managed = marker_path.exists()

        if is_managed:
            managed_runners.append(
                RunnerDir(path=entry, name=entry.name, is_managed=True)
            )

    return managed_runners


def get_runner_listener_process(runner_dir: Path) -> subprocess.Popen[bytes] | None:
    """Get the listener process for a runner directory by executable path.

    The listener process is identified by the executable path under the runner
    directory (e.g., <runner_dir>/run.cmd on Windows or <runner_dir>/run.sh on Unix).
    Process name alone is not used to avoid false positives.

    Args:
        runner_dir: Path to the runner directory

    Returns:
        The Popen object if found, None otherwise
    """
    # Determine the listener executable based on platform
    if sys.platform == "win32":
        listener_exe = runner_dir / "run.cmd"
    else:
        listener_exe = runner_dir / "run.sh"

    if not listener_exe.exists():
        return None

    # Find processes by executable path
    try:
        # Use psutil to find processes by executable path
        import psutil

        for proc in psutil.process_iter(["pid", "exe"]):
            try:
                if proc.info["exe"] and Path(proc.info["exe"]).resolve() == listener_exe.resolve():
                    # Return a Popen-like object (psutil.Process is not Popen, but has similar interface)
                    # We'll use psutil.Process for process management
                    return proc  # type: ignore
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except ImportError:
        # Fallback: use platform-specific process enumeration
        # This is less reliable but better than nothing
        pass

    return None


def mint_remove_token(gh: GitHub) -> str:
    """Mint a remove token for deregistering a runner.

    Args:
        gh: GitHub client instance

    Returns:
        The remove token

    Raises:
        GitHubError: If the GitHub API call fails
    """
    result = gh.run(
        ["api", "-X", "POST", "repos/{owner}/{repo}/actions/runners/remove-token"],
        json_output=True,
    )
    if not result or not isinstance(result, dict):
        raise GitHubError("Failed to mint remove token: invalid response")
    token = result.get("token")
    if not token or not isinstance(token, str):
        raise GitHubError("Failed to mint remove token: no token in response")
    return token


def remove_runner(
    runner_dir: Path,
    remove_token: str,
    *,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Remove a runner using the config.cmd script with the remove token.

    Args:
        runner_dir: Path to the runner directory
        remove_token: The remove token from GitHub
        dry_run: If True, print what would be done without executing

    Returns:
        Tuple of (success, error_message)
    """
    config_cmd = runner_dir / "config.cmd"
    if not config_cmd.exists():
        return False, f"config.cmd not found in {runner_dir}"

    if dry_run:
        return True, f"Would run: {config_cmd} remove --token {remove_token}"

    try:
        result = subprocess.run(
            [str(config_cmd), "remove", "--token", remove_token],
            cwd=runner_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return False, f"config.cmd remove failed: {result.stderr}"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "config.cmd remove timed out"
    except Exception as e:
        return False, f"config.cmd remove failed: {e}"


def stop_runner_process(process: subprocess.Popen[bytes] | None, *, dry_run: bool = False) -> tuple[bool, str]:
    """Stop a runner listener process gracefully.

    Args:
        process: The process to stop (psutil.Process or Popen)
        dry_run: If True, print what would be done without executing

    Returns:
        Tuple of (success, error_message)
    """
    if process is None:
        return True, ""

    if dry_run:
        return True, f"Would stop process {process.pid}"

    try:
        import psutil

        # Convert to psutil.Process if it's a Popen
        if not isinstance(process, psutil.Process):
            proc = psutil.Process(process.pid)
        else:
            proc = process

        # Try graceful termination first
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except psutil.TimeoutExpired:
            # Force kill if graceful termination fails
            proc.kill()
            proc.wait(timeout=5)

        return True, ""
    except ImportError:
        # Fallback for systems without psutil
        try:
            process.terminate()
            process.wait(timeout=10)
            return True, ""
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            return True, ""
    except Exception as e:
        return False, f"Failed to stop process: {e}"


def delete_runner_dir(runner_dir: Path, *, dry_run: bool = False) -> tuple[bool, str]:
    """Delete a runner directory.

    Args:
        runner_dir: Path to the runner directory
        dry_run: If True, print what would be done without executing

    Returns:
        Tuple of (success, error_message)
    """
    if not runner_dir.exists():
        return True, ""

    if dry_run:
        return True, f"Would delete directory: {runner_dir}"

    try:
        import shutil

        shutil.rmtree(runner_dir)
        return True, ""
    except Exception as e:
        return False, f"Failed to delete directory: {e}"


def gracefully_remove_runner(
    runner_dir: RunnerDir,
    gh: GitHub,
    *,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Gracefully remove a runner: drain, deregister, stop process, delete directory.

    Sequence:
    1. Check runner is not busy (via GitHub API)
    2. Mint remove token
    3. Run config.cmd remove --token
    4. Stop listener process
    5. Delete directory

    Every step is gated on the charlie-managed marker. If any step fails,
    the error is returned and the runner is left in a diagnosable state.

    Args:
        runner_dir: The runner directory to remove
        gh: GitHub client instance
        dry_run: If True, print what would be done without executing

    Returns:
        Tuple of (success, error_message)
    """
    if not runner_dir.is_managed:
        return False, f"Runner {runner_dir.name} is not charlie-managed (no marker file)"

    # Step 1: Check runner is not busy
    runners_data = gh.run(["api", "repos/{owner}/{repo}/actions/runners"], json_output=True)
    runners = runners_data.get("runners", []) if runners_data else []
    runner_entry = next((r for r in runners if r.get("name") == runner_dir.name), None)
    if runner_entry and runner_entry.get("busy"):
        return False, f"Runner {runner_dir.name} is busy, cannot remove"

    # Step 2: Mint remove token
    try:
        remove_token = mint_remove_token(gh)
    except GitHubError as e:
        return False, f"Failed to mint remove token: {e}"

    # Step 3: Run config.cmd remove
    success, error = remove_runner(runner_dir.path, remove_token, dry_run=dry_run)
    if not success:
        return False, f"config.cmd remove failed: {error}"

    # Step 4: Stop listener process
    process = get_runner_listener_process(runner_dir.path)
    success, error = stop_runner_process(process, dry_run=dry_run)
    if not success:
        return False, f"Failed to stop process: {error}"

    # Step 5: Delete directory
    success, error = delete_runner_dir(runner_dir.path, dry_run=dry_run)
    if not success:
        return False, f"Failed to delete directory: {error}"

    return True, ""


def get_last_scale_event_time(state_dir: Path) -> datetime | None:
    """Get the timestamp of the last scale event (up or down).

    Args:
        state_dir: The state directory (e.g., .var/charlie-work)

    Returns:
        The timestamp of the last scale event, or None if no event recorded
    """
    scale_event_path = state_dir / "runner-scale-event.json"
    if not scale_event_path.exists():
        return None

    try:
        with scale_event_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            timestamp_str = data.get("timestamp")
            if timestamp_str:
                return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return None


def record_scale_event(state_dir: Path, event_type: str) -> None:
    """Record a scale event (up or down) for cooldown tracking.

    Uses atomic temp-file + replace pattern.

    Args:
        state_dir: The state directory (e.g., .var/charlie-work)
        event_type: The type of scale event ("up" or "down")
    """
    scale_event_path = state_dir / "runner-scale-event.json"
    scale_event_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event_type": event_type,
    }

    tmp_path = scale_event_path.with_suffix(scale_event_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    tmp_path.replace(scale_event_path)


def is_in_cooldown(state_dir: Path, cooldown_minutes: int) -> bool:
    """Check if we are in the cooldown period after a scale event.

    Args:
        state_dir: The state directory (e.g., .var/charlie-work)
        cooldown_minutes: Cooldown period in minutes

    Returns:
        True if in cooldown, False otherwise
    """
    last_event = get_last_scale_event_time(state_dir)
    if last_event is None:
        return False

    elapsed = (datetime.now(UTC) - last_event).total_seconds() / 60
    return elapsed < cooldown_minutes


def scale_down_idle_runners(
    managed_root: Path,
    runner_dir_prefix: str,
    gh: GitHub,
    config: RunnerScalingConfig,
    state_dir: Path,
    *,
    dry_run: bool = False,
) -> tuple[int, list[str]]:
    """Scale down idle runners by gracefully removing them.

    Conditions for scale-down:
    - Pool has been idle for at least idle_scale_down_minutes
    - Not in cooldown period
    - Runner is not busy
    - Runner is charlie-managed
    - Runner count is above min_runners

    Args:
        managed_root: Root directory where runner instances are managed
        runner_dir_prefix: Directory name prefix for runner instances
        gh: GitHub client instance
        config: Runner scaling configuration
        state_dir: The state directory (e.g., .var/charlie-work)
        dry_run: If True, print what would be done without executing

    Returns:
        Tuple of (number_of_runners_removed, list_of_error_messages)
    """
    # Check cooldown
    if is_in_cooldown(state_dir, config.cooldown_minutes):
        return 0, ["In cooldown period"]

    # Check idle duration
    if not is_pool_idle_for_minutes(state_dir, config.idle_scale_down_minutes):
        return 0, ["Pool not idle for required duration"]

    # Discover managed runners
    managed_runners = discover_managed_runners(managed_root, runner_dir_prefix)

    # Check min_runners floor
    if len(managed_runners) <= config.min_runners:
        return 0, ["At min_runners floor"]

    # Select one runner to remove (oldest by directory name)
    # We only remove one at a time to be conservative
    runners_to_remove = sorted(managed_runners, key=lambda r: r.name)[:1]

    removed_count = 0
    errors: list[str] = []

    for runner_dir in runners_to_remove:
        success, error = gracefully_remove_runner(runner_dir, gh, dry_run=dry_run)
        if success:
            removed_count += 1
        else:
            errors.append(f"Failed to remove {runner_dir.name}: {error}")

    # Record scale event if we removed any runners
    if removed_count > 0:
        record_scale_event(state_dir, "down")

    return removed_count, errors


def ensure_runner_running(
    runner_dir: RunnerDir,
    config: RunnerScalingConfig,
    *,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Ensure a configured runner is running by relaunching it if necessary.

    This is the recovery path for managed runners that die on reboot/logoff.
    Managed runners are plain processes (not services), so they need to be
    restarted via ensure-started.

    Args:
        runner_dir: The runner directory to check and potentially start
        config: Runner scaling configuration
        dry_run: If True, print what would be done without executing

    Returns:
        Tuple of (success, error_message)
    """
    if not runner_dir.is_managed:
        return False, f"Runner {runner_dir.name} is not charlie-managed (no marker file)"

    # Check if the runner is already running by process path
    process = get_runner_listener_process(runner_dir.path)
    if process is not None:
        return True, f"Runner {runner_dir.name} is already running"

    # Runner is not running, relaunch it
    # Determine the launch command based on platform
    if sys.platform == "win32":
        launch_script = runner_dir.path / "run.cmd"
    else:
        launch_script = runner_dir.path / "run.sh"

    if not launch_script.exists():
        return False, f"Launch script not found: {launch_script}"

    if dry_run:
        return True, f"Would launch runner: {launch_script}"

    try:
        # Launch the runner in a decontaminated environment
        # We use subprocess.Popen with detached flags to run as a background process
        if sys.platform == "win32":
            # Windows: use DETACHED_PROCESS and CREATE_NEW_PROCESS_GROUP
            startupinfo = subprocess.STARTUPINFO()  # type: ignore
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore
            startupinfo.wShowWindow = subprocess.SW_HIDE  # type: ignore
            proc = subprocess.Popen(
                [str(launch_script)],
                cwd=runner_dir.path,
                startupinfo=startupinfo,  # type: ignore
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,  # type: ignore
            )
        else:
            # Unix: use double-fork to daemonize
            proc = subprocess.Popen(
                [str(launch_script)],
                cwd=runner_dir.path,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        return True, f"Launched runner {runner_dir.name} with PID {proc.pid}"
    except Exception as e:
        return False, f"Failed to launch runner: {e}"


def ensure_runners_started(
    managed_root: Path,
    runner_dir_prefix: str,
    config: RunnerScalingConfig,
    *,
    dry_run: bool = False,
) -> tuple[int, list[str]]:
    """Ensure all configured managed runners are running.

    Relaunches any configured-but-not-running managed runner (decontaminated env).
    This replaces the ad-hoc start-runners.ps1 loop.

    Args:
        managed_root: Root directory where runner instances are managed
        runner_dir_prefix: Directory name prefix for runner instances
        config: Runner scaling configuration
        dry_run: If True, print what would be done without executing

    Returns:
        Tuple of (number_of_runners_started, list_of_status_messages)
    """
    # Discover managed runners
    managed_runners = discover_managed_runners(managed_root, runner_dir_prefix)

    started_count = 0
    messages: list[str] = []

    for runner_dir in managed_runners:
        success, message = ensure_runner_running(runner_dir, config, dry_run=dry_run)
        if success and "already running" not in message:
            started_count += 1
        messages.append(f"{runner_dir.name}: {message}")

    return started_count, messages


def observe_runner_pool(
    gh: GitHub,
    config: RunnerScalingConfig,
    *,
    workflow_filename: str | None = None,
    default_branch: str | None = None,
    state_dir: Path | None = None,
) -> RunnerPoolState:
    """Observe runner pool state for a repository.

    This is a read-only function that collects observability data:
    - Queries GitHub Actions runners via gh API
    - Queries CI workflow run queue depth
    - Samples host CPU and RAM metrics via psutil
    - Derives pressure classification
    - Optionally saves a pool sample for idle detection if state_dir is provided

    Args:
        gh: GitHub client instance
        config: Runner scaling configuration
        workflow_filename: Optional CI workflow filename to filter queue depth.
            If None, counts runs on the repository's default branch instead.
        default_branch: Optional default branch name for the queue-depth query.
            Only consulted when ``workflow_filename`` is None. If also None, the
            branch is resolved once via the repository metadata endpoint.
        state_dir: Optional state directory path for persisting pool samples.
            If provided, a pool sample is saved for idle detection.

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

    # Save pool sample for idle detection if state_dir is provided
    if state_dir is not None:
        sample = PoolSample(
            timestamp=timestamp,
            busy=busy_runners > 0,
            queued_jobs=queued_jobs,
        )
        save_pool_sample(state_dir, sample)
        # Periodically clean up old samples (every call is fine, it's cheap)
        cleanup_pool_samples(state_dir, max_age_minutes=60)

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
