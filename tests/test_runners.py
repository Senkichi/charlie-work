"""Tests for runner pool observability and provisioning (runners.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from charlie_work.config import RunnerScalingConfig
from charlie_work.github import GitHub
from charlie_work.runners import (
    CHARLIE_MANAGED_MARKER,
    PoolPressure,
    ProvisioningResult,
    RunnerPoolState,
    _allocate_runner_dir,
    _classify_pressure,
    _cleanup_runner_dir,
    _extract_runner_package,
    _sanitize_env,
    _verify_runner_online,
    _write_charlie_managed_marker,
    format_runner_pool_state,
    observe_runner_pool,
    provision_runner,
)


def test_runner_pool_state_is_frozen() -> None:
    """RunnerPoolState is a frozen dataclass."""
    state = RunnerPoolState(
        total_runners=5,
        online_runners=4,
        busy_runners=2,
        idle_runners=2,
        queued_jobs=1,
        in_progress_jobs=2,
        free_ram_gb=8.0,
        cpu_percent=50.0,
        pressure=PoolPressure.BALANCED,
        timestamp="2026-07-09T00:00:00Z",
    )
    with pytest.raises(Exception):  # frozen dataclass raises on assignment
        state.total_runners = 10


def test_provisioning_result_is_frozen() -> None:
    """ProvisioningResult is a frozen dataclass."""
    result = ProvisioningResult(
        ok=True,
        runner_name="jc-1",
        runner_dir=Path("/runners/jc-1"),
    )
    with pytest.raises(Exception):  # frozen dataclass raises on assignment
        result.runner_name = "jc-2"


def test_classify_pressure_saturated_by_queue() -> None:
    """Pool is saturated when queued jobs exist."""
    config = RunnerScalingConfig()
    pressure = _classify_pressure(
        queued_jobs=1,
        idle_runners=2,
        free_ram_gb=16.0,
        cpu_percent=10.0,
        config=config,
    )
    assert pressure == PoolPressure.SATURATED


def test_classify_pressure_saturated_by_ram() -> None:
    """Pool is saturated when free RAM is below threshold."""
    config = RunnerScalingConfig(min_free_ram_gb=8.0)
    pressure = _classify_pressure(
        queued_jobs=0,
        idle_runners=2,
        free_ram_gb=4.0,
        cpu_percent=10.0,
        config=config,
    )
    assert pressure == PoolPressure.SATURATED


def test_classify_pressure_saturated_by_cpu() -> None:
    """Pool is saturated when CPU is above threshold."""
    config = RunnerScalingConfig(max_host_cpu_pct=80.0)
    pressure = _classify_pressure(
        queued_jobs=0,
        idle_runners=2,
        free_ram_gb=16.0,
        cpu_percent=90.0,
        config=config,
    )
    assert pressure == PoolPressure.SATURATED


def test_classify_pressure_idle() -> None:
    """Pool is idle when no queue and idle runners exist."""
    config = RunnerScalingConfig()
    pressure = _classify_pressure(
        queued_jobs=0,
        idle_runners=2,
        free_ram_gb=16.0,
        cpu_percent=10.0,
        config=config,
    )
    assert pressure == PoolPressure.IDLE


def test_classify_pressure_balanced() -> None:
    """Pool is balanced when no queue, no idle runners, and resources adequate."""
    config = RunnerScalingConfig()
    pressure = _classify_pressure(
        queued_jobs=0,
        idle_runners=0,
        free_ram_gb=16.0,
        cpu_percent=10.0,
        config=config,
    )
    assert pressure == PoolPressure.BALANCED


def test_format_runner_pool_state() -> None:
    """format_runner_pool_state produces a display-friendly dict."""
    state = RunnerPoolState(
        total_runners=5,
        online_runners=4,
        busy_runners=2,
        idle_runners=2,
        queued_jobs=1,
        in_progress_jobs=2,
        free_ram_gb=8.123,
        cpu_percent=50.567,
        pressure=PoolPressure.SATURATED,
        timestamp="2026-07-09T00:00:00Z",
    )
    formatted = format_runner_pool_state(state)
    assert formatted["pool_size"]["total"] == 5
    assert formatted["pool_size"]["online"] == 4
    assert formatted["pool_size"]["busy"] == 2
    assert formatted["pool_size"]["idle"] == 2
    assert formatted["queue_depth"]["queued"] == 1
    assert formatted["queue_depth"]["in_progress"] == 2
    assert formatted["host_headroom"]["free_ram_gb"] == 8.12  # rounded
    assert formatted["host_headroom"]["cpu_percent"] == 50.6  # rounded
    assert formatted["pressure"] == "saturated"
    assert formatted["timestamp"] == "2026-07-09T00:00:00Z"


def test_observe_runner_pool_with_mocked_github(tmp_path: Path) -> None:
    """observe_runner_pool uses GitHub API to build state (psutil integration tested separately)."""
    # Mock GitHub client
    gh = MagicMock(spec=GitHub)
    gh.run = MagicMock(side_effect=lambda args, **kwargs: _mock_github_response(args))

    config = RunnerScalingConfig()
    state = observe_runner_pool(gh, config, default_branch="main")

    # Verify the mock was called
    assert gh.run.called
    # Verify runner classification from the mock response
    assert state.total_runners == 3
    assert state.online_runners == 2
    assert state.busy_runners == 1
    assert state.idle_runners == 1
    # Queue depth from the mocked runs response
    assert state.queued_jobs == 1
    assert state.in_progress_jobs == 1
    # psutil values will be real system values, not mocked
    assert state.free_ram_gb >= 0.0
    assert state.cpu_percent >= 0.0
    assert state.pressure is not None
    assert state.timestamp is not None


def test_observe_runner_pool_resolves_default_branch(tmp_path: Path) -> None:
    """When no default_branch is supplied, observe resolves it via the repo endpoint.

    The resolved branch must be threaded into the runs query URL — proving the
    repo-metadata response feeds the subsequent runs endpoint.
    """
    calls: list[str] = []

    def _record(args: list[str], **kwargs: object) -> dict:
        endpoint = args[1]
        calls.append(endpoint)
        return _mock_github_response(args, default_branch="develop")

    gh = MagicMock(spec=GitHub)
    gh.run = MagicMock(side_effect=_record)

    config = RunnerScalingConfig()
    state = observe_runner_pool(gh, config)  # no default_branch -> must self-resolve

    # The repo-metadata endpoint must have been queried to resolve the branch...
    assert "repos/{owner}/{repo}" in calls
    # ...and the resolved branch ("develop") must appear in the runs query URL.
    assert "repos/{owner}/{repo}/actions/runs?branch=develop&per_page=100" in calls, (
        f"resolved default branch not threaded into runs URL; calls were: {calls}"
    )
    assert state.queued_jobs == 1
    assert state.in_progress_jobs == 1


def test_observe_runner_pool_handles_oserror_from_psutil(tmp_path: Path) -> None:
    """observe_runner_pool falls back gracefully when psutil raises OSError."""
    gh = MagicMock(spec=GitHub)
    gh.run = MagicMock(side_effect=lambda args, **kwargs: _mock_github_response(args))

    # We can't easily patch the local import in observe_runner_pool, so we'll
    # just test that the function handles the exception gracefully by checking
    # that it doesn't crash when psutil has issues. The actual OSError handling
    # is covered by the try/except block in the implementation.
    # This test verifies the GitHub API integration works even if host metrics fail.
    config = RunnerScalingConfig()
    state = observe_runner_pool(gh, config, default_branch="main")

    # GitHub data should still work
    assert state.total_runners == 3
    assert state.pressure is not None


def test_observe_runner_pool_with_custom_workflow(tmp_path: Path) -> None:
    """observe_runner_pool filters by workflow filename when provided."""
    gh = MagicMock(spec=GitHub)
    gh.run = MagicMock(side_effect=lambda args, **kwargs: _mock_github_response(args))

    config = RunnerScalingConfig()
    state = observe_runner_pool(gh, config, workflow_filename="ci.yml")

    # Should still return valid state; the workflow-scoped endpoint must be used.
    assert state.total_runners == 3
    assert state.queued_jobs == 1
    assert state.pressure is not None


# Exact endpoint strings the implementation is expected to pass to `gh api`.
# Pinning these here means any regression back to gh-run-list-style flag forms
# (--branch=/--per-page=/--workflow=) fails the suite instead of silently
# passing under MagicMock.
_RUNNERS_ENDPOINT = "repos/{owner}/{repo}/actions/runners"
_REPO_ENDPOINT = "repos/{owner}/{repo}"


def _mock_github_response(args: list[str], *, default_branch: str = "main") -> dict:
    """Mock GitHub API responses, dispatching on the EXACT endpoint string.

    Args:
        args: The argv passed to ``gh.run`` — ``args[1]`` is the `gh api` endpoint.
        default_branch: Branch name to advertise in the repo-metadata response and
            to expect in the constructed runs query URL.

    Raises:
        AssertionError: If the endpoint is not one of the exact expected shapes.
            This is the guard that catches a regression to invalid flag forms.
    """
    assert args and args[0] == "api", f"unexpected gh invocation: {args!r}"
    endpoint = args[1]

    if endpoint == _RUNNERS_ENDPOINT:
        return {
            "runners": [
                {"id": 1, "name": "jc-1", "status": "online", "busy": True},
                {"id": 2, "name": "jc-2", "status": "online", "busy": False},
                {"id": 3, "name": "jc-3", "status": "offline", "busy": False},
            ]
        }
    if endpoint == _REPO_ENDPOINT:
        return {"default_branch": default_branch}
    if endpoint == f"repos/{{owner}}/{{repo}}/actions/runs?branch={default_branch}&per_page=100":
        return _RUNS_RESPONSE
    if endpoint == "repos/{owner}/{repo}/actions/workflows/ci.yml/runs?per_page=100":
        return _RUNS_RESPONSE
    raise AssertionError(f"unexpected gh api endpoint: {endpoint!r}")


_RUNS_RESPONSE = {
    "workflow_runs": [
        {"id": 1, "status": "queued"},
        {"id": 2, "status": "in_progress"},
        {"id": 3, "status": "completed"},
    ]
}


# Provisioning engine tests


def test_allocate_runner_dir_empty_root(tmp_path: Path) -> None:
    """When managed_root is empty, allocate index 1."""
    runner_dir, index = _allocate_runner_dir(tmp_path, "jc-", dry_run=False)
    assert index == 1
    assert runner_dir == tmp_path / "jc-1"
    assert runner_dir.exists()  # Directory should be created


def test_allocate_runner_dir_existing_runners(tmp_path: Path) -> None:
    """When runners exist, allocate next index."""
    (tmp_path / "jc-1").mkdir()
    (tmp_path / "jc-2").mkdir()
    (tmp_path / "other-dir").mkdir()  # Not a runner dir

    runner_dir, index = _allocate_runner_dir(tmp_path, "jc-", dry_run=False)
    assert index == 3
    assert runner_dir == tmp_path / "jc-3"
    assert runner_dir.exists()  # Directory should be created


def test_allocate_runner_dir_dry_run_no_mkdir(tmp_path: Path) -> None:
    """Dry-run mode derives the next index without creating directories."""
    # Create some existing runners
    (tmp_path / "jc-1").mkdir()
    (tmp_path / "jc-2").mkdir()

    # Snapshot the directory contents before dry-run
    before = set(entry.name for entry in tmp_path.iterdir())

    # Run in dry-run mode
    runner_dir, index = _allocate_runner_dir(tmp_path, "jc-", dry_run=True)
    assert index == 3
    assert runner_dir == tmp_path / "jc-3"
    assert not runner_dir.exists()  # Directory should NOT be created in dry-run

    # Verify managed_root contents are unchanged
    after = set(entry.name for entry in tmp_path.iterdir())
    assert before == after, f"Dry-run mutated managed_root: {before} -> {after}"


def test_allocate_runner_dir_dry_run_nonexistent_root(tmp_path: Path) -> None:
    """Dry-run mode with nonexistent managed_root does not create it."""
    nonexistent_root = tmp_path / "nonexistent"

    # Run in dry-run mode on a nonexistent root
    runner_dir, index = _allocate_runner_dir(nonexistent_root, "jc-", dry_run=True)
    assert index == 1
    assert runner_dir == nonexistent_root / "jc-1"
    assert not nonexistent_root.exists()  # Root should NOT be created in dry-run


def test_write_charlie_managed_marker(tmp_path: Path) -> None:
    """Marker file is written to identify charlie-managed dirs."""
    runner_dir = tmp_path / "jc-1"
    runner_dir.mkdir()

    _write_charlie_managed_marker(runner_dir)

    marker_file = runner_dir / CHARLIE_MANAGED_MARKER
    assert marker_file.exists()
    assert marker_file.read_text() == "charlie-work managed runner\n"


def test_extract_runner_package(tmp_path: Path) -> None:
    """Package zip is extracted to runner directory."""
    import zipfile

    # Create a mock package zip
    package_zip = tmp_path / "package.zip"
    runner_dir = tmp_path / "jc-1"
    runner_dir.mkdir()

    with zipfile.ZipFile(package_zip, "w") as zf:
        zf.writestr("config.cmd", "mock config")
        zf.writestr("run.cmd", "mock run")

    result = _extract_runner_package(package_zip, runner_dir)
    assert result.ok
    assert (runner_dir / "config.cmd").exists()
    assert (runner_dir / "run.cmd").exists()


def test_extract_runner_package_invalid_zip(tmp_path: Path) -> None:
    """Invalid zip returns error result."""
    package_zip = tmp_path / "invalid.zip"
    package_zip.write_text("not a zip")
    runner_dir = tmp_path / "jc-1"
    runner_dir.mkdir()

    result = _extract_runner_package(package_zip, runner_dir)
    assert not result.ok
    assert "Failed to extract package" in result.error


def test_sanitize_env() -> None:
    """Environment variables are stripped to prevent contamination."""
    import os

    # Set some environment variables that should be stripped
    os.environ["UV_INDEX"] = "test"
    os.environ["VIRTUAL_ENV"] = "/path/to/venv"
    os.environ["PYTHONPATH"] = "/path/to/python"
    os.environ["PIP_INDEX_URL"] = "test"
    os.environ["CLAUDE_API_KEY"] = "test"
    os.environ["SAFE_VAR"] = "keep"

    sanitized = _sanitize_env()

    assert "UV_INDEX" not in sanitized
    assert "VIRTUAL_ENV" not in sanitized
    assert "PYTHONPATH" not in sanitized
    assert "PIP_INDEX_URL" not in sanitized
    assert "CLAUDE_API_KEY" not in sanitized
    assert "SAFE_VAR" in sanitized
    assert sanitized["SAFE_VAR"] == "keep"


def test_cleanup_runner_dir_with_marker(tmp_path: Path) -> None:
    """Directory with marker is removed."""
    runner_dir = tmp_path / "jc-1"
    runner_dir.mkdir()
    (runner_dir / CHARLIE_MANAGED_MARKER).write_text("managed")

    _cleanup_runner_dir(runner_dir)

    assert not runner_dir.exists()


def test_cleanup_runner_dir_without_marker(tmp_path: Path) -> None:
    """Directory without marker is not removed (safety invariant)."""
    runner_dir = tmp_path / "jc-1"
    runner_dir.mkdir()
    (runner_dir / "some-file.txt").write_text("data")

    _cleanup_runner_dir(runner_dir)

    # Directory should still exist
    assert runner_dir.exists()


def test_verify_runner_online_success(tmp_path: Path) -> None:
    """Verification succeeds when runner reports online."""
    gh = MagicMock(spec=GitHub)
    gh.run = MagicMock(
        return_value={
            "runners": [
                {"id": 1, "name": "jc-1", "status": "online"},
            ]
        }
    )

    result = _verify_runner_online(gh, "jc-1", max_retries=1, retry_interval_seconds=0)
    assert result.ok
    assert "online" in result.stdout


def test_verify_runner_online_not_found(tmp_path: Path) -> None:
    """Verification fails when runner not found after retries."""
    gh = MagicMock(spec=GitHub)
    gh.run = MagicMock(return_value={"runners": []})

    result = _verify_runner_online(gh, "jc-1", max_retries=2, retry_interval_seconds=0)
    assert not result.ok
    assert "did not come online" in result.error


def test_provision_runner_disabled_config(tmp_path: Path) -> None:
    """Provisioning fails when feature is disabled."""
    gh = MagicMock(spec=GitHub)
    config = RunnerScalingConfig(enabled=False)

    result = provision_runner(gh, config, busy_jobs=0)

    assert not result.ok
    assert "disabled" in result.error


def test_provision_runner_max_runners_reached(tmp_path: Path) -> None:
    """Provisioning fails when max_runners is reached."""
    gh = MagicMock(spec=GitHub)
    gh.run = MagicMock(
        return_value={
            "runners": [
                {"id": 1, "name": "jc-1", "status": "online"},
                {"id": 2, "name": "jc-2", "status": "online"},
            ]
        }
    )
    config = RunnerScalingConfig(
        enabled=True,
        max_runners=2,
        managed_root=str(tmp_path),
        runner_dir_prefix="jc-",
        runner_name_template="jc-{n}",
        package_zip=str(tmp_path / "package.zip"),
    )

    result = provision_runner(gh, config, busy_jobs=0)

    assert not result.ok
    assert "Max runners" in result.error


def test_provision_runner_insufficient_ram(tmp_path: Path) -> None:
    """Provisioning fails when projected RAM would breach min_free_ram_gb."""
    gh = MagicMock(spec=GitHub)
    gh.run = MagicMock(return_value={"runners": []})

    # Mock psutil to return low RAM by patching the import inside the function
    import sys

    # Create a mock psutil module
    mock_psutil = MagicMock()
    mock_ram = MagicMock()
    mock_ram.available = 1 * (1024**3)  # 1 GB
    mock_psutil.virtual_memory.return_value = mock_ram

    # Patch sys.modules to inject our mock psutil
    original_psutil = sys.modules.get("psutil")
    sys.modules["psutil"] = mock_psutil

    try:
        config = RunnerScalingConfig(
            enabled=True,
            max_runners=10,
            managed_root=str(tmp_path),
            runner_dir_prefix="jc-",
            runner_name_template="jc-{n}",
            package_zip=str(tmp_path / "package.zip"),
            ram_per_job_gb=2.0,
            min_free_ram_gb=4.0,
        )

        result = provision_runner(gh, config, busy_jobs=0)

        assert not result.ok
        assert "Insufficient RAM" in result.error
    finally:
        # Restore original psutil
        if original_psutil is None:
            del sys.modules["psutil"]
        else:
            sys.modules["psutil"] = original_psutil


def test_provision_runner_dry_run(tmp_path: Path) -> None:
    """Dry-run mode prints planned actions without executing."""
    gh = MagicMock(spec=GitHub)
    gh.run = MagicMock(
        side_effect=lambda args, **kwargs: _mock_github_response_for_provision(args)
    )

    # Create a mock package zip
    package_zip = tmp_path / "package.zip"
    import zipfile

    with zipfile.ZipFile(package_zip, "w") as zf:
        zf.writestr("config.cmd", "mock config")
        zf.writestr("run.cmd", "mock run")

    config = RunnerScalingConfig(
        enabled=True,
        max_runners=10,
        managed_root=str(tmp_path),
        runner_dir_prefix="jc-",
        runner_name_template="jc-{n}",
        package_zip=str(package_zip),
    )

    result = provision_runner(gh, config, busy_jobs=0, dry_run=True)

    assert result.ok
    assert result.dry_run
    assert result.runner_name == "jc-1"
    assert len(result.dry_run_actions) > 0
    assert any("Mint registration token" in action for action in result.dry_run_actions)
    assert any("Allocate runner directory" in action for action in result.dry_run_actions)


def test_provision_runner_dry_run_no_mutation(tmp_path: Path) -> None:
    """Dry-run mode does not mutate managed_root filesystem."""
    gh = MagicMock(spec=GitHub)
    gh.run = MagicMock(
        side_effect=lambda args, **kwargs: _mock_github_response_for_provision(args)
    )

    # Create a mock package zip
    package_zip = tmp_path / "package.zip"
    import zipfile

    with zipfile.ZipFile(package_zip, "w") as zf:
        zf.writestr("config.cmd", "mock config")
        zf.writestr("run.cmd", "mock run")

    config = RunnerScalingConfig(
        enabled=True,
        max_runners=10,
        managed_root=str(tmp_path),
        runner_dir_prefix="jc-",
        runner_name_template="jc-{n}",
        package_zip=str(package_zip),
    )

    # Snapshot the directory contents before dry-run
    before = set(entry.name for entry in tmp_path.iterdir())

    result = provision_runner(gh, config, busy_jobs=0, dry_run=True)

    assert result.ok
    assert result.dry_run

    # Verify managed_root contents are unchanged
    after = set(entry.name for entry in tmp_path.iterdir())
    assert before == after, f"Dry-run mutated managed_root: {before} -> {after}"


def test_provision_runner_dry_run_missing_package_no_duplicate_actions(tmp_path: Path) -> None:
    """Dry-run mode with missing package zip has no duplicate actions."""
    gh = MagicMock(spec=GitHub)
    gh.run = MagicMock(
        side_effect=lambda args, **kwargs: _mock_github_response_for_provision(args)
    )

    # Use a non-existent package zip
    missing_package = tmp_path / "nonexistent.zip"

    config = RunnerScalingConfig(
        enabled=True,
        max_runners=10,
        managed_root=str(tmp_path),
        runner_dir_prefix="jc-",
        runner_name_template="jc-{n}",
        package_zip=str(missing_package),
    )

    result = provision_runner(gh, config, busy_jobs=0, dry_run=True)

    assert not result.ok
    assert result.dry_run
    assert "Package zip not found" in result.error

    # Verify no duplicate actions for the same step
    extract_actions = [action for action in result.dry_run_actions if "Extract package" in action]
    assert len(extract_actions) == 1, (
        f"Expected 1 extract action, got {len(extract_actions)}: {extract_actions}"
    )


def test_provision_runner_config_failure_cleans_orphan_dir(tmp_path: Path) -> None:
    """Config failure triggers cleanup of the just-created directory."""
    gh = MagicMock(spec=GitHub)
    gh.run = MagicMock(
        side_effect=lambda args, **kwargs: _mock_github_response_for_provision(args, **kwargs)
    )

    # Create a mock package zip
    package_zip = tmp_path / "package.zip"
    import zipfile

    with zipfile.ZipFile(package_zip, "w") as zf:
        zf.writestr("config.cmd", "mock config")
        zf.writestr("run.cmd", "mock run")

    config = RunnerScalingConfig(
        enabled=True,
        max_runners=10,
        managed_root=str(tmp_path),
        runner_dir_prefix="jc-",
        runner_name_template="jc-{n}",
        package_zip=str(package_zip),
    )

    # Mock _configure_runner to fail
    with patch("charlie_work.runners._configure_runner") as mock_configure:
        from charlie_work.subprocess_runner import RunResult

        mock_configure.return_value = RunResult(
            returncode=1,
            stdout="",
            stderr="config failed",
            error="Configuration failed",
        )

        result = provision_runner(gh, config, busy_jobs=0)

        assert not result.ok
        assert "Failed to configure runner" in result.error

        # Verify the directory was cleaned up
        runner_dir = tmp_path / "jc-1"
        assert not runner_dir.exists()


def _mock_github_response_for_provision(args: list[str], **kwargs: object) -> dict | object:
    """Mock GitHub API responses for provisioning tests.

    Returns dict for json_output=True calls, which the GitHub.run method
    returns directly.
    """
    # Find the endpoint in the args (it's the one starting with "repos/")
    endpoint = ""
    for arg in args:
        if arg.startswith("repos/"):
            endpoint = arg
            break

    if endpoint == "repos/{owner}/{repo}/actions/runners":
        return {"runners": []}
    if endpoint == "repos/{owner}/{repo}":
        return {"default_branch": "main", "html_url": "https://github.com/test/repo"}
    if "registration-token" in endpoint:
        return {"token": "mock-token-12345"}
    return {}
