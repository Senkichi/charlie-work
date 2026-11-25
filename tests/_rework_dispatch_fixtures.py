"""Shared fakes/helpers for the rework-dispatch test modules.

Hoisted verbatim out of ``tests/test_charlie_work.py`` (issue #1547, Track-1
wave 1/8) when the rework/dry-run dispatch and infra-blocked routing seams
were split into seam-named siblings -- the ``tests/_*.py`` hoisted-fixture
convention is the sanctioned import target for shared test helpers (see
``tests/test_zero_cross_test_import_guard.py``). ``test_charlie_work.py``
itself imports the four helpers its remaining tests still use.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from typing import (
    Any,
    Callable,
)
from _fakes_github import (
    FakeGitHub,
    FakeGitHubWithChecksAndAnnotations,
)
from charlie_work.adapters import SessionDispatchResult
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.host.fakes import FakeWorkerLauncher
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp
from charlie_work.write_gate import WriteGate


# Issue #1264 (W6 PR2): the WriteGate must carry THIS test's own state_file
# as state_path -- WriteGate.save_state() writes to self.state_path, not to
# whatever path the converted function was also given.
def _wg(state_file: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo="charlie-work")


class _FakeGitHubWithInfraBlockedJob(FakeGitHubWithChecksAndAnnotations):
    """FakeGitHub whose ``actions_job`` returns a configurable per-job-id mapping.

    Simulates the Actions API response for a budget-exhausted / runner-outage
    job: zero steps, FAILURE conclusion, optionally a billing annotation.
    """

    def __init__(
        self,
        checks: list[dict[str, Any]] | None = None,
        annotations_by_check_run_id: dict[int, list[dict[str, Any]]] | None = None,
        jobs_by_check_run_id: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(checks, annotations_by_check_run_id)
        self._jobs = jobs_by_check_run_id or {}

    def actions_job(self, job_id: int) -> dict[str, Any] | None:
        return self._jobs.get(job_id)


def _dispatch_rework_config() -> OrchestratorConfig:
    return OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )


def _normalize_ws(text: str) -> str:
    """Collapse whitespace (including the template's hard line-wraps) so
    substring assertions don't depend on exact word-wrap columns."""
    return " ".join(text.split())


def _init_repo_with_remote_inline(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bare origin remote + local clone with one commit on main.

    Inlined here (instead of importing from test_worktree.py) so this test
    module stays self-contained for the salvage-push regression tests.
    Returns ``(remote, repo_root)``.

    Mirrors ``test_worktree._init_repo(bare=True)``: a bare repo cannot
    receive commits directly, so a temporary non-bare repo is initialized
    with ``--initial-branch=main``, seeded with one commit, then cloned
    with ``--bare`` to produce the remote.
    """
    import shutil

    # Build a temp non-bare repo with one commit on main, then clone --bare.
    temp_repo = tmp_path / "remote-temp"
    temp_repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=temp_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=temp_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=temp_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (temp_repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=temp_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=temp_repo,
        check=True,
        capture_output=True,
        text=True,
    )

    remote = tmp_path / "remote"
    subprocess.run(
        ["git", "clone", "--bare", str(temp_repo), str(remote)],
        check=True,
        capture_output=True,
        text=True,
    )
    shutil.rmtree(temp_repo, ignore_errors=True)

    repo_root = tmp_path / "repo"
    subprocess.run(
        ["git", "clone", str(remote), str(repo_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return remote, repo_root


def _api_worker_config_for_test(
    *,
    enabled: bool = True,
    provider_name: str = "kimi-k3",
    api_key_env: str = "MOONSHOT_API_KEY",
) -> Any:
    """Build an ApiWorkerConfig for rescue-tier combined-manifest tests."""
    from charlie_work.config import ApiBudgetConfig, ApiProviderConfig, ApiWorkerConfig

    provider = ApiProviderConfig(
        base_url="https://api.moonshot.ai/anthropic",
        api_key_env=api_key_env,
        model="kimi-k3",
        input_usd_per_mtok=3.0,
        output_usd_per_mtok=15.0,
        cached_input_usd_per_mtok=0.30,
    )
    return ApiWorkerConfig(
        enabled=enabled,
        provider=provider_name,
        max_concurrent_sessions=1,
        providers={provider_name: provider},
        budget=ApiBudgetConfig(),
        worker_template="worker_claude_code.md",
        rework_template="rework.md",
    )


def _fake_dispatch_sessions_writing_manifests(
    manifest_writes: list[str], tmp_path: Path
) -> Callable[..., list[Any]]:
    """Build a fake dispatch_sessions that writes manifest+results like the real
    one, recording the adapter label each call used. This lets a rework test
    verify the combined trailing write's label without launching workers."""

    def _fake(_repo_root, manifest_path, results_path, settings, requests):
        from charlie_work.adapters import (
            SessionDispatchResult,
            write_session_manifest,
            write_session_results,
        )

        write_session_manifest(manifest_path, requests, adapter=settings.adapter)
        manifest_writes.append(settings.adapter)
        results = [
            SessionDispatchResult(
                issue_number=r.issue_number,
                issue_title=r.issue_title,
                prompt_path=str(r.prompt_path),
                branch_name=r.branch_name,
                adapter=settings.adapter,
                ok=True,
                pid=4242,
                process_start_time=1.0,
            )
            for r in requests
        ]
        write_session_results(results_path, results)
        return results

    return _fake


def _seed_two_rework_issues(paths, config: Any, *, rescue_issue_numbers: set[int]) -> None:
    """Seed state with two rework_requested issues (123 normal, 124 rescue-marked)
    and open PRs, plus rework prompts on disk."""
    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {"number": 123, "status": "rework_requested"}
        state["issues"]["124"] = {"number": 124, "status": "rework_requested"}
        state["prs"]["456"] = {"number": 456, "issue_number": 123}
        pr124_fields: dict[str, Any] = {"number": 457, "issue_number": 124}
        if 124 in rescue_issue_numbers:
            pr124_fields["rescue_attempted"] = True
            pr124_fields["rescue_cause"] = "rework_cycle_cap"
        state["prs"]["457"] = pr124_fields
        save_state(paths.state_file, state)
    for pr_num in (456, 457):
        pr_dir = paths.prs / f"pr-{pr_num}"
        pr_dir.mkdir(parents=True, exist_ok=True)
        (pr_dir / "rework-prompt.md").write_text("rework prompt", encoding="utf-8")


class _TwoReworkIssuesGitHub(FakeGitHub):
    """FakeGitHub with two issues (123, 124) and two open PRs (456, 457)."""

    def __init__(self) -> None:
        super().__init__()
        self.issues.append(
            {
                "number": 124,
                "title": "Another issue",
                "url": "https://example.test/issues/124",
                "body": "Body",
                "labels": [{"name": "agent:needs-rework"}],
                "state": "OPEN",
            }
        )
        self.prs.append(
            {
                "number": 457,
                "title": "Fix #124",
                "url": "https://example.test/pull/457",
                "headRefName": "agent/issue-124-another-issue",
                "baseRefName": "main",
                "headRefOid": "sha-def456",
                "mergeStateStatus": "CLEAN",
                "body": "Closes #124",
                "labels": [],
                "isCrossRepository": False,
                "state": "OPEN",
            }
        )


def _blocked_env_timestamps(n: int) -> list[str]:
    """``n`` recent ISO timestamps for pre-seeding ``blocked_environment_at``."""
    return [
        (datetime.now(UTC) - timedelta(minutes=i)).isoformat().replace("+00:00", "Z")
        for i in range(n)
    ]


def _rework_launch_at_redispatch_cap_app(
    tmp_path: Path, fake_host: Callable[..., Any], *, pid: int, process_start_time: float
) -> tuple[OrchestratorApp, FakeWorkerLauncher]:
    """A ``dispatch_rework`` fixture whose next launch succeeds and trips the cap.

    Issue 123 is ``rework_requested`` with ``redispatch_at`` already at
    ``max_auto_redispatch`` and no recorded ``reviewed_head_sha``, so the
    pre-launch no-op check has nothing to compare and the launch goes ahead;
    counting that launch puts the issue one over the cap. The record also
    carries a previous dispatch epoch's Worker facts and prompt path, so a
    test can tell this launch's values from leftovers. The scripted launcher
    (installed as the ``worker_launch`` Host port) reports ``pid`` and
    ``process_start_time`` for every request and logs the requests it saw.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
            "redispatch_at": [now_iso, now_iso],
            "worker_pid": 11111,
            "worker_process_start_time": 1.0,
            "prompt_path": "previous-epoch/rework-prompt.md",
        }
        save_state(paths.state_file, state)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "rework-prompt.md").write_text("Fix the issues", encoding="utf-8")

    def launched(_repo_root, _manifest, _results, settings, requests):
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter=settings.adapter,
                ok=True,
                pid=pid,
                process_start_time=process_start_time,
            )
            for request in requests
        ]

    launcher = FakeWorkerLauncher([launched])
    fake_host(worker_launch=launcher)
    return OrchestratorApp(tmp_path, paths, config, ReworkGitHub()), launcher
