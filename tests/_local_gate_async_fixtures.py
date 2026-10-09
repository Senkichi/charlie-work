"""Shared fixtures/helpers for the local merge gate's async-suite tests.

Extracted from ``test_local_merge_gate_async.py`` (#2127) so the infra-outcome
tests can reuse them -- test modules must never import each other.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from charlie_work import local_suite_runner
from charlie_work.config import OrchestratorConfig, build_config_from_data
from charlie_work.labels import transition
from charlie_work.local_lane import branch_head_sha
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

# A pid no supported OS can allocate (Linux ``pid_max`` caps at 2**22; the
# Windows cid table cannot reach 2**30): deterministically dead on every pass
# and, unlike a just-exited pid, unrecyclable -- a real dead pid can be
# reissued to an unrelated process inside the assertion window, and a claim
# carrying ``local_suite_process_start_time=None`` gives the gate's liveness
# check no fingerprint to disambiguate it with (#2207: bare pids are not
# unique across dead processes).
UNALLOCATABLE_PID = 1 << 30


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


def _drop_claim(app: OrchestratorApp, pr_number: int) -> None:
    """Lose the claim (supervisor restart / re-arm) while the result file stays."""
    # Local import: ``local_merge_gate`` does ``import charlie_work.workflow``
    # at module level, so it may only load after workflow is fully built.
    from charlie_work.orchestration import local_merge_gate

    # The result lands just before the runner exits; a still-live pid file would
    # make the gate defer instead of launching.
    _wait_pid_dead(
        int(load_state_locked(app.paths.state_file)["prs"][str(pr_number)]["local_suite_pid"])
    )
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        record = state["prs"][str(pr_number)]
        for field in local_merge_gate.LOCAL_SUITE_CLAIM_FIELDS:
            record.pop(field, None)
        save_state(app.paths.state_file, state)


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


def _seed_dead_claim(app: OrchestratorApp, repo_root: Path, number: int, head: str) -> None:
    """Persist a claim whose runner is deterministically dead (``UNALLOCATABLE_PID``,
    no fingerprint), as left behind by a wrapper that exited without reporting."""
    paths = _gate_paths(app, number)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"][str(number)].update(
            {
                "local_suite_pid": UNALLOCATABLE_PID,
                "local_suite_process_start_time": None,
                "local_suite_started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "local_suite_log": str(paths.log),
                "local_suite_gate_dir": str(paths.gate_dir),
                "local_suite_head": head,
                "local_suite_base_sha": branch_head_sha(repo_root, "main"),
            }
        )
        save_state(app.paths.state_file, state)


def _event_kinds(app: OrchestratorApp) -> list[str]:
    return [e["kind"] for e in load_state_locked(app.paths.state_file)["events"]]


def _events_of_kind(app: OrchestratorApp, kind: str) -> list[dict]:
    return [e for e in load_state_locked(app.paths.state_file)["events"] if e["kind"] == kind]


@pytest.fixture(autouse=True)
def _no_windows_child_enumeration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``kill_process_tree``'s child enumeration off the PowerShell path.

    Every gate kill in the importing module -- the in-flight timeout, plus
    each ``_kill_claimed_gate`` teardown -- enumerates the suite's children
    for the ``killed_pids`` report, and on Windows that enumeration is one
    ``Get-CimInstance Win32_Process`` spawn whose latency swings from ~0.3 s
    to its 5 s timeout under host load. That bimodal ~4.5 s per kill is the
    ledger-flagged regression (#2597; #2642 flagged the same cost inside the
    async module's measured call phases). ``taskkill /T`` still fells the
    real tree; the enumeration only feeds a report these tests never assert
    on. Autouse, so importing the name arms it for the importing module.
    """
    import charlie_work.process_utils as process_utils

    monkeypatch.setattr(process_utils, "_enumerate_child_pids", lambda _pid: [])


@pytest.fixture(name="moved_base_gate")
def _moved_base_gate(lane_repo: Path) -> Iterator[OrchestratorApp]:
    """Completed-gate scaffold charged to setup: green suite, dropped claim, moved base.

    The reuse keying under test (#2125) needs a *finished* green gate whose
    base then moved; the real suite run (``_wait_for_result`` polls a detached
    interpreter plus its suite child) and the ~11 git/process spawns of lane
    acquisition drift with shared-runner load, none of it the property under
    test -- the property is the *second* pass relaunching instead of reusing.
    Paying it inside ``call`` is what the ledger flagged as a 2.1x regression
    (#2662; same fix shape as #2642/#2624/#2634). The kill of the second,
    in-flight suite lands in teardown -- a ``finally`` in the body would
    count as ``call``. Requests the importing module's ``lane_repo`` fixture.
    """
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)
    assert app._local_merge_approved()[0]["outcome"] == "suite_launched"
    _wait_for_result(app, 7)
    _drop_claim(app, 7)
    _commit_file(lane_repo, "other.py", "x = 1\n", "another PR merged first")
    yield app
    # Tolerant of a cleared claim -- the kill is a no-op when the test
    # escalated or the suite resolved on its own.
    _kill_claimed_gate(app, 7)


# Explicit fixture name (the ``_wt_scratch`` pattern): importing modules bring
# the private symbol into their namespace so pytest registers the fixture; a
# matching function name would collide with the fixture parameter in test
# signatures (ruff F811).
@pytest.fixture(name="approved_sleep_gate")
def _approved_sleep_gate(lane_repo: Path) -> Iterator[OrchestratorApp]:
    """Lane scaffold charged to setup: repo, branch, app, approved record.

    None of the acquisition is the property under test, and its per-spawn
    git/process cost drifts with shared-runner load -- paying it inside the
    measured ``call`` phase is what the ledger flagged as a 2.4x regression
    (#2642; same fix shape as #2624/#2634). The call phase keeps only the
    launch pass plus its assertions, and the in-flight suite's kill lands in
    teardown instead of the test's ``finally`` (which still counts as
    ``call``). Requests the importing module's ``lane_repo`` fixture.
    """
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    config = _lane_config(lane_repo, issues_dir, dispatch={"test_command": SLEEP_SUITE})
    app = _lane_app(lane_repo, issues_dir, config=config)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)
    yield app
    # Tolerant of a cleared claim -- the kill is a no-op when the test
    # escalated or the suite resolved on its own.
    _kill_claimed_gate(app, 7)
