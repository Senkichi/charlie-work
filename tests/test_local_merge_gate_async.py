"""Issue #1974: the local merge gate launches the full suite asynchronously.

Before the fix, ``_local_merge_approved`` ran ``run_full_suite`` synchronously
inside the supervisor pass -- 33-47 minutes per approved record froze every
fleet lane. The gate is now a two-phase state machine:

* pass N   -- base-sync merge inline (fast), then ``Popen`` the suite runner
  (``python -m charlie_work.local_suite_runner``) and return; the claim
  (``local_suite_pid``/``local_suite_started_at``/``local_suite_log``/head+base
  SHAs) is persisted on the record;
* pass N+k -- read ``suite-result.json``/live-pid state, advance the base on
  green, route to rework on red/timeout, re-sync + relaunch when the base
  moved, and relaunch (bounded) when the runner died without reporting.

At most one gate is in flight per repository; other approved records get
``local_merge_deferred``. Every test below drives a real git repo, a real
``LocalFileGitHub`` backend, and real runner processes (``python -c`` suites),
matching the existing ``test_local_lane.py`` / ``test_local_issues_merge_gate``
fixture style.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from _git_leak_guard import scrubbed_git_env
from charlie_work import local_suite_runner, quiesce
from charlie_work.config import OrchestratorConfig, build_config_from_data
from charlie_work.host_load import pytest_tree_load
from charlie_work.labels import transition
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.local_lane import (
    SUITE_TIMEOUT_SECONDS,
    branch_head_sha,
    is_ancestor,
)
from charlie_work.paths import runtime_paths
from charlie_work.process_utils import is_pid_alive, kill_process_tree
from charlie_work.state import load_state, load_state_locked, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

# The gate delegate does ``import charlie_work.workflow as _wf`` at module
# level, so it may only be imported *after* ``charlie_work.workflow`` has been
# fully imported -- otherwise delegate discovery sees it partially initialized
# and raises (the #1798 hazard; same pattern as
# ``test_issue_1314_operator_queue_followups``).
from charlie_work.orchestration.local_merge_gate import (  # noqa: E402
    LOCAL_SUITE_GATE_MAX_ORPHANS,
    LOCAL_SUITE_GATE_MAX_RESYNCS,
)


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # The scrub stops ambient GIT_DIR/GIT_CONFIG_* redirecting commands at the
    # outer repo; the session's isolation vars (#2060) must survive it.
    env = scrubbed_git_env()
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


# Suite commands: ``python -c`` bodies keep every test hermetic -- no pytest
# collection, no imports, deterministic exits. ``suite_command_argv`` appends
# ``-q --tb=short`` which lands harmlessly in ``sys.argv`` for ``-c``.
PASS_SUITE = 'python -c "pass"'
FAIL_SUITE = 'python -c "import sys; sys.exit(3)"'
SLEEP_SUITE = 'python -c "import time; time.sleep(600)"'


@pytest.fixture
def lane_repo() -> Path:
    """Real git repo under the system temp dir (``$GIT_DIR``-path headroom --
    see ``test_local_issues_merge_gate.lane_repo``)."""
    return Path(tempfile.mkdtemp(prefix="cw-gate-async-"))


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
    transition(app.gh, labels, issue_number, "local_work_ready")
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


def _wait_for_pid_file(app: OrchestratorApp, pr_number: int, timeout_seconds: float = 30) -> dict:
    paths = _gate_paths(app, pr_number)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        meta = local_suite_runner.read_gate_pid(paths)
        if meta is not None:
            return meta
        time.sleep(0.05)
    raise AssertionError(f"suite pid file for pr-{pr_number} never appeared")


def _wait_pid_dead(pid: int, timeout_seconds: float = 15) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not is_pid_alive(pid):
            return
        time.sleep(0.1)
    raise AssertionError(f"pid {pid} still alive after {timeout_seconds}s")


def _kill_claimed_gate(app: OrchestratorApp, pr_number: int) -> int:
    """Kill the in-flight suite tree named by the record's claim; returns pid."""
    state = load_state_locked(app.paths.state_file)
    record = state["prs"][str(pr_number)]
    pid = int(record["local_suite_pid"])
    kill_process_tree(pid, record.get("local_suite_process_start_time"))
    _wait_pid_dead(pid)
    return pid


def _kill_any_claimed_gates(app: OrchestratorApp, *pr_numbers: int) -> None:
    """Kill every suite tree the records claim; tolerant of missing claims.

    Cleanup for the multi-record ordering tests must also cover the buggy
    shape they assert against -- a run that wrongly launched a second suite
    leaves a claim on the lower-numbered record too.
    """
    state = load_state_locked(app.paths.state_file)
    for pr_number in pr_numbers:
        record = state["prs"].get(str(pr_number), {})
        pid = record.get("local_suite_pid")
        if isinstance(pid, int) and pid > 0:
            kill_process_tree(pid, record.get("local_suite_process_start_time"))


def _strip_claim_fields(app: OrchestratorApp, pr_number: int) -> None:
    """Erase the in-flight claim -- simulates a crash between the wrapper's
    Popen and the claim write, or any state loss that orphaned the runner."""
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        record = state["prs"][str(pr_number)]
        for field in (
            "local_suite_pid",
            "local_suite_process_start_time",
            "local_suite_started_at",
            "local_suite_log",
            "local_suite_gate_dir",
            "local_suite_head",
            "local_suite_base_sha",
            "local_suite_argv",
            "local_suite_resync_count",
            "local_suite_orphan_count",
        ):
            record.pop(field, None)
        save_state(app.paths.state_file, state)


def _backdate_gate_start(app: OrchestratorApp, pr_number: int) -> None:
    """Age the in-flight claim past ``SUITE_TIMEOUT_SECONDS``."""
    stale = (
        (datetime.now(UTC) - timedelta(seconds=SUITE_TIMEOUT_SECONDS + 60))
        .isoformat()
        .replace("+00:00", "Z")
    )
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"][str(pr_number)]["local_suite_started_at"] = stale
        save_state(app.paths.state_file, state)


def _dead_pid() -> int:
    """A pid that was real a moment ago and is now definitely exited."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=30)
    return child.pid


def _event_kinds(app: OrchestratorApp) -> list[str]:
    return [e["kind"] for e in load_state_locked(app.paths.state_file)["events"]]


def _events_of_kind(app: OrchestratorApp, kind: str) -> list[dict]:
    return [e for e in load_state_locked(app.paths.state_file)["events"] if e["kind"] == kind]


# ---------------------------------------------------------------------------
# AC: "A fleet pass with an approved local record returns in seconds, not
# suite-duration. Test: a fake suite command that sleeps, and assert that the
# pass returns before it exits."
# ---------------------------------------------------------------------------


def test_approved_record_launches_suite_and_pass_returns_immediately(
    lane_repo: Path,
) -> None:
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    config = _lane_config(lane_repo, issues_dir, dispatch={"test_command": SLEEP_SUITE})
    app = _lane_app(lane_repo, issues_dir, config=config)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)

    t0 = time.monotonic()
    results = app._local_merge_approved()
    elapsed = time.monotonic() - t0

    try:
        assert elapsed < 60, f"gate pass blocked for {elapsed:.1f}s on a 600s suite"
        assert results[0]["outcome"] == "suite_launched"
        state = load_state_locked(app.paths.state_file)
        record = state["prs"]["7"]
        pid = int(record["local_suite_pid"])
        assert is_pid_alive(pid)
        assert record["local_suite_started_at"]
        assert record["local_suite_log"].endswith("suite.log")
        assert record["local_suite_head"] == branch_head_sha(lane_repo, "agent/issue-7-x")
        launched = _events_of_kind(app, "local_suite_launched")
        assert len(launched) == 1
        assert launched[0]["payload"]["pid"] == pid
        # The suite is genuinely still running -- the pass did not wait for it.
        assert is_pid_alive(pid)
    finally:
        _kill_claimed_gate(app, 7)


def test_green_result_advances_base_on_next_pass(lane_repo: Path) -> None:
    """AC: a green suite result on a later pass advances the base."""
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)

    results = app._local_merge_approved()
    assert results[0]["outcome"] == "suite_launched"

    result = _wait_for_result(app, 7)
    assert result["ok"] is True

    results = app._local_merge_approved()

    assert results[0]["outcome"] in ("merged", "already_merged")
    assert is_ancestor(lane_repo, head, branch_head_sha(lane_repo, "main"))
    state = load_state_locked(app.paths.state_file)
    record = state["prs"]["7"]
    assert record["status"] == "merged"
    assert record.get("local_suite_pid") is None
    assert state["issues"]["7"]["status"] == "closed"
    kinds = _event_kinds(app)
    assert "local_suite_launched" in kinds
    assert "local_suite_ok" in kinds
    assert "local_suite_result" in kinds
    assert "merge_succeeded" in kinds


def test_red_result_routes_to_rework(lane_repo: Path) -> None:
    """AC: a red suite result on a later pass routes to rework."""
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    config = _lane_config(lane_repo, issues_dir, dispatch={"test_command": FAIL_SUITE})
    app = _lane_app(lane_repo, issues_dir, config=config)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)

    results = app._local_merge_approved()
    assert results[0]["outcome"] == "suite_launched"
    result = _wait_for_result(app, 7)
    assert result["ok"] is False
    assert result["returncode"] == 3

    results = app._local_merge_approved()

    assert results[0]["outcome"] == "suite_failed"
    assert results[0]["routed_to"] == "rework"
    state = load_state_locked(app.paths.state_file)
    assert state["prs"]["7"]["status"] == "rework_requested"
    assert state["prs"]["7"].get("local_suite_pid") is None
    assert state["issues"]["7"]["status"] == "rework_requested"
    assert not is_ancestor(lane_repo, head, branch_head_sha(lane_repo, "main"))
    kinds = _event_kinds(app)
    assert "local_suite_failed" in kinds
    assert "local_suite_result" in kinds


def test_second_approved_record_deferred_while_gate_in_flight(lane_repo: Path) -> None:
    """AC: at most one gate in flight per repo; others get local_merge_deferred."""
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head7 = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    head8 = _make_branch(lane_repo, "agent/issue-8-y", "b.py", "b = 1\n")
    config = _lane_config(lane_repo, issues_dir, dispatch={"test_command": SLEEP_SUITE})
    app = _lane_app(lane_repo, issues_dir, config=config)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head7)
    _adopt_and_approve(app, issues_dir, 8, "agent/issue-8-y", head8)

    results = app._local_merge_approved()

    try:
        by_issue = {r["issue"]: r for r in results}
        assert by_issue[7]["outcome"] == "suite_launched"
        assert by_issue[8]["outcome"] == "deferred"
        deferred = _events_of_kind(app, "local_merge_deferred")
        assert deferred
        assert deferred[-1]["payload"]["issue_number"] == 8
        assert "in flight" in deferred[-1]["payload"]["detail"]
        state = load_state_locked(app.paths.state_file)
        assert state["prs"]["8"]["status"] == "approved"
        assert state["prs"]["8"].get("local_suite_pid") is None
    finally:
        _kill_claimed_gate(app, 7)


def test_unclaimed_lower_record_defers_while_higher_gate_claimed(lane_repo: Path) -> None:
    """Regression: the one-gate-per-repo bound must not depend on iteration
    order. A higher-numbered record already holds a live claimed suite; the
    walk reaches the lower-numbered unclaimed record first, so a flag that
    only fills in as the walk proceeds would launch a second concurrent
    suite. PR 7 must defer, not launch."""
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head7 = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    head9 = _make_branch(lane_repo, "agent/issue-9-z", "c.py", "c = 1\n")
    config = _lane_config(lane_repo, issues_dir, dispatch={"test_command": SLEEP_SUITE})
    app = _lane_app(lane_repo, issues_dir, config=config)
    # Only the higher-numbered record is approved for the first pass, so it
    # owns the in-flight gate; the lower-numbered record is approved after.
    _adopt_and_approve(app, issues_dir, 9, "agent/issue-9-z", head9)
    first = app._local_merge_approved()
    assert first[0]["outcome"] == "suite_launched"
    pid9 = int(load_state_locked(app.paths.state_file)["prs"]["9"]["local_suite_pid"])
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head7)

    results = app._local_merge_approved()

    try:
        by_issue = {r["issue"]: r for r in results}
        assert by_issue[7]["outcome"] == "deferred"
        assert by_issue[9]["outcome"] == "suite_running"
        deferred = _events_of_kind(app, "local_merge_deferred")
        assert deferred
        assert deferred[-1]["payload"]["issue_number"] == 7
        assert "in flight" in deferred[-1]["payload"]["detail"]
        state = load_state_locked(app.paths.state_file)
        # No second runner pid or claim for the deferred record, and PR 9's
        # suite is still the only one running.
        assert state["prs"]["7"]["status"] == "approved"
        assert state["prs"]["7"].get("local_suite_pid") is None
        assert int(state["prs"]["9"]["local_suite_pid"]) == pid9
        assert is_pid_alive(pid9)
        assert len(_events_of_kind(app, "local_suite_launched")) == 1
    finally:
        _kill_any_claimed_gates(app, 7, 9)


def test_unclaimed_lower_record_defers_while_higher_runner_adopted(lane_repo: Path) -> None:
    """Same ordering bound through the crash-recovery path: a live
    ``suite-runner.json`` on a higher-numbered record whose claim write never
    landed still holds the gate -- the adopt step re-attaches it and the
    lower-numbered unclaimed record defers rather than launching beside it."""
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head7 = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    head9 = _make_branch(lane_repo, "agent/issue-9-z", "c.py", "c = 1\n")
    config = _lane_config(lane_repo, issues_dir, dispatch={"test_command": SLEEP_SUITE})
    app = _lane_app(lane_repo, issues_dir, config=config)
    _adopt_and_approve(app, issues_dir, 9, "agent/issue-9-z", head9)
    app._local_merge_approved()
    meta = _wait_for_pid_file(app, 9)
    pid9 = int(meta["pid"])
    _strip_claim_fields(app, 9)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head7)

    results = app._local_merge_approved()

    try:
        by_issue = {r["issue"]: r for r in results}
        assert by_issue[7]["outcome"] == "deferred"
        assert by_issue[9]["outcome"] == "suite_running"
        assert by_issue[9]["adopted"] is True
        state = load_state_locked(app.paths.state_file)
        assert int(state["prs"]["9"]["local_suite_pid"]) == pid9
        assert state["prs"]["7"].get("local_suite_pid") is None
        assert len(_events_of_kind(app, "local_suite_launched")) == 1
        assert is_pid_alive(pid9)
    finally:
        kill_process_tree(pid9)
        _kill_any_claimed_gates(app, 7, 9)


def test_restart_mid_suite_resumes_no_double_launch(lane_repo: Path) -> None:
    """AC: a supervisor restart mid-suite resumes from persisted state and
    never runs the suite twice at once."""
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    config = _lane_config(lane_repo, issues_dir, dispatch={"test_command": SLEEP_SUITE})
    app = _lane_app(lane_repo, issues_dir, config=config)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)

    app._local_merge_approved()
    pid = int(load_state_locked(app.paths.state_file)["prs"]["7"]["local_suite_pid"])

    try:
        # "Restart": a fresh OrchestratorApp on the same state -- the only
        # continuity is the persisted claim plus the live runner process.
        restarted = _lane_app(lane_repo, issues_dir, config=config)
        results = restarted._local_merge_approved()

        assert results[0]["outcome"] == "suite_running"
        state = load_state_locked(app.paths.state_file)
        assert int(state["prs"]["7"]["local_suite_pid"]) == pid
        assert is_pid_alive(pid)
        assert len(_events_of_kind(app, "local_suite_launched")) == 1
    finally:
        _kill_claimed_gate(app, 7)


def test_lost_claim_adopts_live_runner_via_pid_file(lane_repo: Path) -> None:
    """Crash window: the wrapper outlived a claim write that never landed (or
    was lost). The pid file it wrote re-attaches the gate instead of launching
    a second suite."""
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    config = _lane_config(lane_repo, issues_dir, dispatch={"test_command": SLEEP_SUITE})
    app = _lane_app(lane_repo, issues_dir, config=config)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)

    app._local_merge_approved()
    meta = _wait_for_pid_file(app, 7)
    pid = int(meta["pid"])
    _strip_claim_fields(app, 7)

    try:
        results = app._local_merge_approved()

        assert results[0]["outcome"] == "suite_running"
        record = load_state_locked(app.paths.state_file)["prs"]["7"]
        assert int(record["local_suite_pid"]) == pid
        assert record["local_suite_head"] == meta["head_sha"]
        assert len(_events_of_kind(app, "local_suite_launched")) == 1
        assert is_pid_alive(pid)
    finally:
        kill_process_tree(pid)


def test_base_moved_during_suite_resyncs_and_relaunches(lane_repo: Path) -> None:
    """AC: base-moved-during-suite causes a re-sync and relaunch -- the merge
    must never land a head that wasn't tested against the current base."""
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)

    app._local_merge_approved()
    first = _wait_for_result(app, 7)
    assert first["ok"] is True
    # The base advances AFTER the suite under test finished.
    _commit_file(lane_repo, "base_new.py", "n = 1\n", "base moved during suite")
    moved_base = branch_head_sha(lane_repo, "main")

    results = app._local_merge_approved()

    assert results[0]["outcome"] == "suite_launched"
    record = load_state_locked(app.paths.state_file)["prs"]["7"]
    assert record["local_suite_base_sha"] == moved_base
    assert record["local_suite_resync_count"] == 1
    assert not is_ancestor(lane_repo, head, branch_head_sha(lane_repo, "main"))

    # The relaunched suite runs green against the resynced base; the next
    # pass is free to merge.
    second = _wait_for_result(app, 7)
    assert second["ok"] is True
    results = app._local_merge_approved()
    assert results[0]["outcome"] in ("merged", "already_merged")
    assert is_ancestor(lane_repo, head, branch_head_sha(lane_repo, "main"))


def test_timeout_kills_tree_and_routes_to_rework(lane_repo: Path) -> None:
    """AC: runtime beyond SUITE_TIMEOUT_SECONDS kills the process tree and
    routes to rework."""
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    config = _lane_config(lane_repo, issues_dir, dispatch={"test_command": SLEEP_SUITE})
    app = _lane_app(lane_repo, issues_dir, config=config)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)

    app._local_merge_approved()
    pid = int(load_state_locked(app.paths.state_file)["prs"]["7"]["local_suite_pid"])
    _backdate_gate_start(app, 7)

    results = app._local_merge_approved()

    assert results[0]["outcome"] == "suite_failed"
    _wait_pid_dead(pid)
    state = load_state_locked(app.paths.state_file)
    assert state["prs"]["7"]["status"] == "rework_requested"
    assert state["prs"]["7"].get("local_suite_pid") is None
    kinds = _event_kinds(app)
    assert "local_suite_failed" in kinds


def test_dead_pid_missing_result_relaunches_bounded_then_escalates(
    lane_repo: Path,
) -> None:
    """A runner that died without writing a result is never green: the gate
    relaunches (bounded), then escalates the infrastructure anomaly."""
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    config = _lane_config(lane_repo, issues_dir, dispatch={"test_command": SLEEP_SUITE})
    app = _lane_app(lane_repo, issues_dir, config=config)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)

    app._local_merge_approved()
    launched_pids = {int(load_state_locked(app.paths.state_file)["prs"]["7"]["local_suite_pid"])}

    for orphan in range(1, LOCAL_SUITE_GATE_MAX_ORPHANS + 1):
        _kill_claimed_gate(app, 7)
        results = app._local_merge_approved()
        assert results[0]["outcome"] == "suite_launched", f"orphan {orphan}"
        state = load_state_locked(app.paths.state_file)
        pid = int(state["prs"]["7"]["local_suite_pid"])
        assert pid not in launched_pids
        launched_pids.add(pid)
        assert state["prs"]["7"]["local_suite_orphan_count"] == orphan
        # Never green on a missing result.
        assert state["prs"]["7"]["status"] == "approved"

    _kill_claimed_gate(app, 7)
    results = app._local_merge_approved()

    assert results[0]["outcome"] == "error"
    state = load_state_locked(app.paths.state_file)
    assert state["prs"]["7"]["status"] == "escalated"
    assert state["prs"]["7"]["escalation_reason"] == "local_merge_error"
    assert "local_merge_failed" in _event_kinds(app)


def test_malformed_result_file_is_never_green(lane_repo: Path) -> None:
    """A result file the gate cannot parse is not a green suite -- the record
    stays approved and the gate relaunches rather than merging unverified."""
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)
    paths = _gate_paths(app, 7)
    paths.gate_dir.mkdir(parents=True, exist_ok=True)
    paths.result.write_text("{not json", encoding="utf-8")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["7"].update(
            {
                "local_suite_pid": _dead_pid(),
                "local_suite_process_start_time": None,
                "local_suite_started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "local_suite_log": str(paths.log),
                "local_suite_gate_dir": str(paths.gate_dir),
                "local_suite_head": head,
                "local_suite_base_sha": branch_head_sha(lane_repo, "main"),
            }
        )
        save_state(app.paths.state_file, state)

    results = app._local_merge_approved()

    try:
        assert results[0]["outcome"] == "suite_launched"
        state = load_state_locked(app.paths.state_file)
        assert state["prs"]["7"]["status"] == "approved"
        assert not is_ancestor(lane_repo, head, branch_head_sha(lane_repo, "main"))
        # The stale file was cleared by the relaunch; a real pid now claims.
        assert is_pid_alive(int(state["prs"]["7"]["local_suite_pid"]))
    finally:
        _kill_claimed_gate(app, 7)


def test_result_for_wrong_head_is_not_accepted(lane_repo: Path) -> None:
    """A result file from an earlier gate epoch (different head under test)
    must not resolve the current claim -- missing-result safety."""
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)
    paths = _gate_paths(app, 7)
    paths.gate_dir.mkdir(parents=True, exist_ok=True)
    # Stale green result stamped for a different head.
    paths.result.write_text(
        '{"ok": true, "returncode": 0, "head_sha": "deadbeef", "argv": []}',
        encoding="utf-8",
    )
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["7"].update(
            {
                "local_suite_pid": _dead_pid(),
                "local_suite_process_start_time": None,
                "local_suite_started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "local_suite_log": str(paths.log),
                "local_suite_gate_dir": str(paths.gate_dir),
                "local_suite_head": head,
                "local_suite_base_sha": branch_head_sha(lane_repo, "main"),
            }
        )
        save_state(app.paths.state_file, state)

    results = app._local_merge_approved()

    try:
        assert results[0]["outcome"] == "suite_launched"
        assert not is_ancestor(lane_repo, head, branch_head_sha(lane_repo, "main"))
    finally:
        _kill_claimed_gate(app, 7)


def test_resync_bound_escalates_when_base_never_settles(lane_repo: Path) -> None:
    """A base that keeps moving during the suite is bounded: after
    ``LOCAL_SUITE_GATE_MAX_RESYNCS`` relaunches the gate escalates instead of
    spinning forever."""
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)

    app._local_merge_approved()
    for resync in range(1, LOCAL_SUITE_GATE_MAX_RESYNCS + 1):
        _wait_for_result(app, 7)
        _commit_file(lane_repo, f"move{resync}.py", "m = 1\n", f"base move {resync}")
        results = app._local_merge_approved()
        assert results[0]["outcome"] == "suite_launched", f"resync {resync}"
        record = load_state_locked(app.paths.state_file)["prs"]["7"]
        assert record["local_suite_resync_count"] == resync

    _wait_for_result(app, 7)
    _commit_file(lane_repo, "move_final.py", "m = 2\n", "base move final")
    results = app._local_merge_approved()

    assert results[0]["outcome"] == "error"
    state = load_state_locked(app.paths.state_file)
    assert state["prs"]["7"]["status"] == "escalated"
    assert state["prs"]["7"]["escalation_reason"] == "local_merge_error"


def test_gate_aborted_when_record_leaves_approved(lane_repo: Path) -> None:
    """An approved record re-routed (operator escalation, packet void) while
    its suite runs must kill the tree and clear the claim -- a stray gate
    must not keep running or falsely resolve later."""
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    config = _lane_config(lane_repo, issues_dir, dispatch={"test_command": SLEEP_SUITE})
    app = _lane_app(lane_repo, issues_dir, config=config)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)

    app._local_merge_approved()
    pid = int(load_state_locked(app.paths.state_file)["prs"]["7"]["local_suite_pid"])
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["7"]["status"] = "escalated"
        save_state(app.paths.state_file, state)

    app._local_merge_approved()

    _wait_pid_dead(pid)
    record = load_state_locked(app.paths.state_file)["prs"]["7"]
    assert record.get("local_suite_pid") is None


def test_gate_runner_tree_is_host_load_attributable(lane_repo: Path) -> None:
    """AC: the host-load governor counts the gate's pytest tree. The wrapper
    argv itself carries the state-dir marker (``--result`` under
    ``.var/charlie-work/dispatches/``) so the tree attributes even when the
    suite's own command line is pathless (``uv run pytest``)."""
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    config = _lane_config(lane_repo, issues_dir)
    app = _lane_app(lane_repo, issues_dir, config=config)
    paths = _gate_paths(app, 7)
    suite_argv = ["uv", "run", "--extra", "dev", "pytest", "-q", "--tb=short"]
    argv = local_suite_runner.gate_runner_argv(
        lane_repo / ".var" / "charlie-work" / "worktrees" / "agent-issue-7-x",
        suite_argv,
        paths=paths,
        head_sha="a" * 40,
        base_sha="b" * 40,
    )
    cmdline = subprocess.list2cmdline(argv)
    procs = [
        quiesce.ProcessInfo(pid=2000, ppid=1, name="python", command_line=cmdline),
        quiesce.ProcessInfo(
            pid=2001,
            ppid=2000,
            name="pytest",
            command_line="uv run --extra dev pytest -q --tb=short",
        ),
    ]

    load = pytest_tree_load(procs, self_pid=999999, scope_paths=[app.paths.dispatches.parent])

    assert load.pytest_tree_count == 1
    assert load.pytest_process_count == 2


def test_suite_log_captures_output_under_dispatch_dir(lane_repo: Path) -> None:
    """The suite's stdout+stderr stream to ``suite.log`` under the lane's
    dispatch dir so a red gate's tail is recoverable for the rework brief."""
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    config = _lane_config(
        lane_repo,
        issues_dir,
        dispatch={"test_command": "python -c \"print('marker-stdout')\""},
    )
    app = _lane_app(lane_repo, issues_dir, config=config)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)

    app._local_merge_approved()
    _wait_for_result(app, 7)

    paths = _gate_paths(app, 7)
    assert paths.log.is_file()
    assert paths.gate_dir.is_relative_to(app.paths.dispatches)
    assert "marker-stdout" in paths.log.read_text(encoding="utf-8", errors="surrogateescape")
