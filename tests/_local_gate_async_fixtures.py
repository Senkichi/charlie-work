"""Shared fixtures/helpers for the local merge gate's async-suite tests.

Extracted from ``test_local_merge_gate_async.py`` (#2127) so the infra-outcome
tests can reuse them -- test modules must never import each other.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from charlie_work import local_suite_runner
from charlie_work.config import OrchestratorConfig, build_config_from_data
from charlie_work.labels import transition
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.paths import runtime_paths
from charlie_work.process_utils import is_pid_alive, kill_process_tree
from charlie_work.state import load_state, load_state_locked, save_state, state_lock
from charlie_work.workflow import OrchestratorApp


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, env=env)
    assert result.returncode == 0, (args, result.stderr)
    return result


def _init_repo(repo_root: Path) -> None:
    """Fresh repo with one commit on ``main`` (same as the sibling fixtures)."""
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init", "--initial-branch=main")
    _git(repo_root, "config", "core.longpaths", "true")
    _git(repo_root, "config", "user.email", "test@example.test")
    _git(repo_root, "config", "user.name", "test")
    _git(repo_root, "commit", "--allow-empty", "-m", "chore: seed")


def _write_issue(
    issues_dir: Path,
    number: int,
    *,
    state: str = "open",
    body: str = "Body.",
    labels: tuple[str, ...] = (),
) -> Path:
    issues_dir.mkdir(parents=True, exist_ok=True)
    labels_yaml = "[" + ", ".join(labels) + "]"
    path = issues_dir / f"{number:03d}_issue.md"
    path.write_text(f"---\nstate: {state}\nlabels: {labels_yaml}\n---\n{body}\n", encoding="utf-8")
    return path


def _commit_file(repo_root: Path, relpath: str, content: str, message: str) -> str:
    path = repo_root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _git(repo_root, "add", relpath)
    _git(repo_root, "commit", "-m", message)
    return _git(repo_root, "rev-parse", "HEAD").stdout.strip()


def _make_branch(repo_root: Path, branch: str, relpath: str, content: str) -> str:
    _git(repo_root, "checkout", "-b", branch)
    head = _commit_file(repo_root, relpath, content, f"feat: {relpath}")
    _git(repo_root, "checkout", "main")
    return head


PASS_SUITE = 'python -c "pass"'

# Prints a pytest-style terminal summary: a real red suite always does, and the
# gate (#2127) treats a summary-less non-zero exit as an infra death instead.
FAIL_SUITE = "python -c \"import sys; print('=== 1 failed, 2 passed in 0.1s ==='); sys.exit(3)\""
# Exits non-zero with no pytest summary at all -- an externally killed run.
KILLED_SUITE = "python -c \"import sys; sys.stdout.write('....... [ 12%]'); sys.exit(1)\""

SLEEP_SUITE = 'python -c "import time; time.sleep(600)"'


def _lane_config(repo_root: Path, issues_dir: Path, **overrides: object) -> OrchestratorConfig:
    data: dict = {
        "local_issues": {"enabled": True, "issues_dir": "docs/issues"},
        "dispatch": {"test_command": PASS_SUITE},
    }
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    return build_config_from_data(data)


def _lane_app(
    repo_root: Path,
    issues_dir: Path,
    config: OrchestratorConfig | None = None,
) -> OrchestratorApp:
    cfg = config or _lane_config(repo_root, issues_dir)
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    paths = runtime_paths(repo_root, cfg.runtime.state_dir)
    return OrchestratorApp(repo_root, paths, cfg, gh)


def _adopt_and_approve(
    app: OrchestratorApp,
    issues_dir: Path,
    issue_number: int,
    branch: str,
    head: str,
) -> None:
    labels = app.config.labels
    _write_issue(issues_dir, issue_number, labels=(labels.ready, labels.in_progress))
    transition(app.gh, labels, issue_number, "local_work_ready", state_path=app.paths.state_file)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state.setdefault("issues", {})[str(issue_number)] = {
            "number": issue_number,
            "title": "Test issue",
            "status": "dispatched",
            "branch_name": branch,
        }
        save_state(app.paths.state_file, state)
    app._local_review_packets()
    result = app.record_local_review(
        issue_number,
        "approved",
        reviewed_head=head,
        verdict_provenance="fresh_llm_review",
    )
    assert result.ok, result.message


def _gate_paths(app: OrchestratorApp, pr_number: int) -> local_suite_runner.SuiteGatePaths:
    return local_suite_runner.suite_gate_paths(app.paths.dispatches, pr_number)


def _wait_for_result(app: OrchestratorApp, pr_number: int, timeout_seconds: float = 90) -> dict:
    """Poll the gate's result file -- the wrapper is a real child process."""
    paths = _gate_paths(app, pr_number)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = local_suite_runner.read_gate_result(paths)
        if result is not None:
            return result
        time.sleep(0.1)
    raise AssertionError(f"suite result for pr-{pr_number} never appeared at {paths.result}")


def _wait_pid_dead(pid: int, timeout_seconds: float = 15) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not is_pid_alive(pid):
            return
        time.sleep(0.1)
    raise AssertionError(f"pid {pid} still alive after {timeout_seconds}s")


def _kill_claimed_gate(app: OrchestratorApp, pr_number: int) -> int:
    """Kill the in-flight suite tree named by the record's claim; returns pid.

    Returns 0 when the claim is already cleared (e.g. the gate escalated), so it
    is safe in a ``finally`` teardown.
    """
    state = load_state_locked(app.paths.state_file)
    record = state["prs"][str(pr_number)]
    if not record.get("local_suite_pid"):
        return 0
    pid = int(record["local_suite_pid"])
    kill_process_tree(pid, record.get("local_suite_process_start_time"))
    _wait_pid_dead(pid)
    return pid


def _gate_identity(app: OrchestratorApp, number: int) -> tuple[int, Any]:
    """Launch identity of the claimed gate: (pid, process start-time fingerprint).

    A bare pid is not unique across dead processes -- the OS may hand a dead
    wrapper's pid to the next launch -- so "a new gate was launched" is asserted
    on the pair, never on the pid alone (#2207).
    """
    record = load_state_locked(app.paths.state_file)["prs"][str(number)]
    return int(record["local_suite_pid"]), record["local_suite_process_start_time"]


def _dead_pid() -> int:
    """A pid that was real a moment ago and is now definitely exited."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=30)
    return child.pid


def _event_kinds(app: OrchestratorApp) -> list[str]:
    return [e["kind"] for e in load_state_locked(app.paths.state_file)["events"]]


def _events_of_kind(app: OrchestratorApp, kind: str) -> list[dict]:
    return [e for e in load_state_locked(app.paths.state_file)["events"] if e["kind"] == kind]
