"""Issue #1923 regression: the state/PID orphan sweep must park a dead
worker's committed local-backend work for review instead of reclaiming the
issue back to ``agent:ready``.

On a ``local_issues`` backend the worker's branch IS the deliverable --
there is no remote to push to and no PR to open. Before this fix,
``_detect_and_handle_orphaned_workers``'s #417 reclaim loop stripped the
active label and re-added ``ready`` unconditionally: the issue went back to
dispatch, the next worker launch died ``worktree_unsafe`` on the dead
worker's own commits, and the issue escalated to the operator queue as if
the worker had failed (the mdls #144 sequence -- 9 issues force-escalated,
~98 downstream issues blocked).

The fix routes the sweep through the same ``_attempt_salvage`` ->
``park_unpublishable_work`` path the sidecar-based dead-worker lane already
uses: a successful park applies ``agent:review-ready`` and advances the
state entry off ``dispatched`` (to ``open_passive``) in one write, so the
``dead_dispatched_reap_minutes`` backstop can never later escalate work
that was already parked.

Covered here:

- happy path: dead dispatched worker + worktree commits ahead -> parked
  (labels, status flip, drift markers cleared, event), no push, and a
  second pass is quiet even with a long-armed drift backstop;
- worktree already reclaimed but branch ref still holds the commits ->
  branch-diff fallback still parks;
- worktree present but zero commits ahead -> normal #417 reclaim (the
  control proving the gate only fires on salvageable work);
- PR-capable backend with identical commits -> unchanged reclaim (the
  capability gate, not the git state, selects the lane);
- park label-write failure -> loud fall-through to reclaim with the
  failure recorded on the event, retryable on the next pass;
- dry-run -> nothing persisted.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

import charlie_work.dead_worker_reap as dead_worker_reap
from _fakes_github import FakeGitHub
from charlie_work.config import (
    ClaudeCodeConfig,
    LabelConfig,
    OrchestratorConfig,
    load_config,
)
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.paths import resolved_layout, runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.worktree import worktree_path_for_branch
from charlie_work.write_gate import WriteGate

# ---------------------------------------------------------------------------
# Helpers -- self-contained per this repo's test-file convention (see
# test_local_review_ready.py / test_salvage_dry_run_1418.py).
# ---------------------------------------------------------------------------


def _git(repo_root: Path, *args: str, cwd: Path | None = None) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd or repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(repo_root: Path) -> None:
    """A fresh git repo with one commit and NO origin remote.

    ``core.longpaths`` is set because pytest's basetemp under this repo's
    ``.var/worker-tmp`` pushes ``.git/objects/<sha>`` paths past the Windows
    MAX_PATH limit; the knob is a no-op elsewhere.
    """
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init", "--initial-branch=main")
    _git(repo_root, "config", "core.longpaths", "true")
    _git(
        repo_root,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "--allow-empty",
        "-m",
        "chore: seed",
    )


def _write_issue(
    issues_dir: Path,
    number: int,
    *,
    slug: str = "issue",
    title: str = "Test issue",
    state: str = "open",
    labels: tuple[str, ...] = (),
) -> Path:
    issues_dir.mkdir(parents=True, exist_ok=True)
    labels_yaml = "[" + ", ".join(labels) + "]"
    path = issues_dir / f"{number:03d}_2026-09-17_{slug}.md"
    path.write_text(
        "---\n"
        f'title: "{title}"\n'
        f"state: {state}\n"
        f"labels: {labels_yaml}\n"
        'created: "2026-09-17"\n'
        'author: "test"\n'
        "---\n"
        "Body text.\n",
        encoding="utf-8",
    )
    return path


def _wg(state_file: Path, *, dry_run: bool = False, repo: str = "test-repo") -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo=repo)


@pytest.fixture
def shallow_wts(tmp_path: Path) -> Iterator[Path]:
    """A worktrees root OUTSIDE ``tmp_path``'s deep nesting.

    pytest's basetemp under this repo's ``.var/worker-tmp`` nests deep
    enough that any worktree path derived from it --
    ``<repo>/.var/charlie-work/worktrees/<branch-slug>`` or even
    ``<tmp_path>/wts/<branch-slug>`` -- trips git's ``$GIT_DIR`` path cap
    on ``git worktree add`` (``fatal: '$GIT_DIR' too big``). A mkdtemp
    under the real %TEMP% keeps the path short. Taking ``tmp_path`` orders
    this teardown BEFORE pytest removes the repo: the worktree dirs are
    plain directories (``git worktree add`` creates no junctions), so a
    recursive delete of the scratch root is safe.
    """
    root = Path(tempfile.mkdtemp(prefix="cwwt1923-"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _local_config(repo_root: Path, worktrees_dir: Path) -> OrchestratorConfig:
    """Config for a local_issues repo whose worktrees live at
    ``worktrees_dir`` -- the ``claude_code.worktrees_dir`` override is the
    config-supported knob production resolves the same way
    (``resolved_layout``), so the sweep still exercises the real
    worktree-path derivation."""
    config_file = repo_root / "orchestrator.config.yaml"
    config_file.write_text(
        "local_issues:\n"
        "  enabled: true\n"
        "claude_code:\n"
        f'  worktrees_dir: "{worktrees_dir.as_posix()}"\n',
        encoding="utf-8",
    )
    return load_config(config_file)


def _add_worktree_commit(repo_root: Path, config: OrchestratorConfig, branch: str) -> Path:
    """Create the worker's worktree at the production-resolved location and
    commit real work on ``branch`` inside it."""
    worktrees_dir = resolved_layout(config, repo_root).worktrees
    worktree_path = worktree_path_for_branch(repo_root, branch, worktrees_dir)
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "worktree", "add", "-b", branch, str(worktree_path))
    (worktree_path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(repo_root, "add", "feature.txt", cwd=worktree_path)
    _git(
        repo_root,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-m",
        "feat: work",
        cwd=worktree_path,
    )
    return worktree_path


def _seed_dead_dispatched(
    state_file: Path,
    issue_number: int,
    branch: str,
    *,
    armed_drift_minutes_ago: int | None = None,
) -> None:
    """State entry for a dead dispatched worker. When
    ``armed_drift_minutes_ago`` is given the entry carries a long-armed
    drift marker -- the exact precondition under which the
    ``dead_dispatched_reap_minutes`` backstop would escalate on this pass
    if the park's status flip were missing."""
    state = load_state(state_file)
    entry = {
        "status": "dispatched",
        "worker_pid": 424242,
        "worker_process_start_time": 1700000000.0,
        "dispatched_at": "2026-09-17T00:00:00Z",
        "branch_name": branch,
    }
    if armed_drift_minutes_ago is not None:
        old = (
            (datetime.now(UTC) - timedelta(minutes=armed_drift_minutes_ago))
            .isoformat()
            .replace("+00:00", "Z")
        )
        entry["orphan_flagged_at"] = old
        entry["orphan_drift_at"] = old
        entry["orphan_drift_fingerprint"] = '{"dead": true}'
    state["issues"][str(issue_number)] = entry
    save_state(state_file, state)


def _run_sweep(
    sessions_dir: Path,
    state_file: Path,
    config: OrchestratorConfig,
    gh,
    write_gate: WriteGate,
    fleet_dir: Path,
) -> None:
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir,
            state_file,
            config,
            gh,
            write_gate=write_gate,
            fleet_dir_override=str(fleet_dir),
        )


def _events(state_file: Path, kind: str, issue_number: int | None = None):
    state = load_state(state_file)
    out = []
    for event in state.get("events", []):
        if event.get("kind") != kind:
            continue
        if (
            issue_number is not None
            and event.get("payload", {}).get("issue_number") != issue_number
        ):
            continue
        out.append(event)
    return out


def _label_names(gh: LocalFileGitHub, issue_number: int) -> set[str]:
    return {entry["name"] for entry in gh.issue_view(issue_number)["labels"]}


# ---------------------------------------------------------------------------
# Happy path: committed work in the dead worker's worktree gets parked.
# ---------------------------------------------------------------------------


def test_orphan_sweep_parks_completed_local_work(tmp_path: Path, shallow_wts: Path) -> None:
    """The #1923 regression itself: a dead dispatched worker whose worktree
    holds commits ahead of the local base is parked ``agent:review-ready``
    with status ``open_passive`` -- never relabeled ``ready`` for a
    redispatch that would die on the worker's own commits.

    The drift markers are pre-seeded 120 minutes old (twice the default
    60-minute ``dead_dispatched_reap_minutes``) so this ALSO proves the
    backstop cannot fire on a parked issue: without the status flip this
    exact pass would emit ``dead_dispatched_worker_reaped`` and escalate.
    """
    labels_cfg = LabelConfig()
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _local_config(repo_root, shallow_wts)

    issue_number = 1923
    branch = "agent/issue-1923-x"
    _add_worktree_commit(repo_root, config, branch)

    issues_dir = repo_root / config.local_issues.issues_dir
    _write_issue(
        issues_dir,
        issue_number,
        title="fix the flaky test",
        labels=(labels_cfg.ready, labels_cfg.in_progress),
    )
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    _seed_dead_dispatched(paths.state_file, issue_number, branch, armed_drift_minutes_ago=120)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    fleet_dir = tmp_path / "fleet"

    with patch.object(dead_worker_reap, "push_branch") as push_mock:
        push_mock.side_effect = AssertionError(
            "push_branch must not be called for a no-PR backend"
        )
        _run_sweep(sessions_dir, paths.state_file, config, gh, _wg(paths.state_file), fleet_dir)

    # Labels: the local_work_ready edge applied -- review_ready added,
    # in_progress removed, ready (not a workflow label) survives.
    current = _label_names(gh, issue_number)
    assert labels_cfg.review_ready in current
    assert labels_cfg.ready in current
    assert labels_cfg.in_progress not in current

    # The branch -- the actual deliverable on a no-remote repo -- survives.
    assert (
        subprocess.run(
            ["git", "rev-parse", "--verify", f"refs/heads/{branch}"],
            cwd=repo_root,
            capture_output=True,
        ).returncode
        == 0
    )

    # State: the entry left the dispatched lane in the same write, the armed
    # drift is cleared, and the #282 liveness fingerprint is preserved for
    # safe worktree cleanup.
    state = load_state(paths.state_file)
    entry = state["issues"][str(issue_number)]
    assert entry["status"] == "open_passive"
    assert entry["worker_pid"] == 424242
    assert "orphan_flagged_at" not in entry
    assert "orphan_drift_at" not in entry
    assert "orphan_drift_fingerprint" not in entry

    # Events: the park is recorded; the reclaim and the backstop never ran.
    ready_events = _events(paths.state_file, "local_work_ready", issue_number)
    assert len(ready_events) == 1
    assert ready_events[0]["payload"]["branch"] == branch
    assert ready_events[0]["payload"]["label_write_ok"] is True
    assert _events(paths.state_file, "session_failed_relabeled", issue_number) == []
    assert _events(paths.state_file, "dead_dispatched_worker_reaped", issue_number) == []

    # The branch comment points the reviewer at the work.
    comments = gh.issue_view(issue_number)["comments"]
    assert len(comments) == 1
    assert branch in comments[0]["body"]

    # Second pass: the entry no longer has status "dispatched", so
    # ``partition_dispatched_by_pid_liveness`` never re-discovers it. The
    # sweep must be fully quiet -- no rediscovery, no drift, no backstop.
    _run_sweep(sessions_dir, paths.state_file, config, gh, _wg(paths.state_file), fleet_dir)
    state = load_state(paths.state_file)
    entry = state["issues"][str(issue_number)]
    assert entry["status"] == "open_passive"
    assert len(_events(paths.state_file, "local_work_ready", issue_number)) == 1
    assert _events(paths.state_file, "session_failed_relabeled", issue_number) == []
    assert _events(paths.state_file, "dead_dispatched_worker_reaped", issue_number) == []
    assert _events(paths.state_file, "orphaned_worker_drift", issue_number) == []


def test_orphan_sweep_parks_when_worktree_gone_but_branch_has_commits(
    tmp_path: Path, shallow_wts: Path
) -> None:
    """The worktree-missing fallback: the worktree may already be reclaimed
    while the branch (the deliverable on a no-remote repo) still carries the
    worker's commits. A non-empty ``branch_diff`` against the local base is
    proof enough to park."""
    labels_cfg = LabelConfig()
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _local_config(repo_root, shallow_wts)

    issue_number = 144
    branch = "agent/issue-144-x"
    worktree_path = _add_worktree_commit(repo_root, config, branch)
    # Simulate the cleanup lane winning the race: worktree gone, branch kept.
    _git(repo_root, "worktree", "remove", "--force", str(worktree_path))

    issues_dir = repo_root / config.local_issues.issues_dir
    _write_issue(
        issues_dir,
        issue_number,
        title="fix the flaky test",
        labels=(labels_cfg.ready, labels_cfg.in_progress),
    )
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    _seed_dead_dispatched(paths.state_file, issue_number, branch)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    _run_sweep(
        sessions_dir, paths.state_file, config, gh, _wg(paths.state_file), tmp_path / "fleet"
    )

    current = _label_names(gh, issue_number)
    assert labels_cfg.review_ready in current
    assert labels_cfg.in_progress not in current
    state = load_state(paths.state_file)
    assert state["issues"][str(issue_number)]["status"] == "open_passive"
    assert len(_events(paths.state_file, "local_work_ready", issue_number)) == 1


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def test_orphan_sweep_reclaims_local_orphan_with_no_commits(
    tmp_path: Path, shallow_wts: Path
) -> None:
    """Control: an identical dead dispatched worker on a local backend whose
    worktree has ZERO commits ahead takes the unchanged #417 reclaim --
    the park gate only fires on provably salvageable work."""
    labels_cfg = LabelConfig()
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _local_config(repo_root, shallow_wts)

    issue_number = 7
    branch = "agent/issue-7-x"
    # Worktree exists but the branch tip equals the base: nothing to park.
    worktrees_dir = resolved_layout(config, repo_root).worktrees
    worktree_path = worktree_path_for_branch(repo_root, branch, worktrees_dir)
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "worktree", "add", "-b", branch, str(worktree_path))

    issues_dir = repo_root / config.local_issues.issues_dir
    _write_issue(
        issues_dir,
        issue_number,
        title="fix the flaky test",
        labels=(labels_cfg.ready, labels_cfg.in_progress),
    )
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    _seed_dead_dispatched(paths.state_file, issue_number, branch)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    _run_sweep(
        sessions_dir, paths.state_file, config, gh, _wg(paths.state_file), tmp_path / "fleet"
    )

    # The plain reclaim ran: active label stripped, ready kept, no
    # review-ready, status still dispatched (requeue for a fresh attempt).
    current = _label_names(gh, issue_number)
    assert labels_cfg.in_progress not in current
    assert labels_cfg.ready in current
    assert labels_cfg.review_ready not in current

    state = load_state(paths.state_file)
    assert state["issues"][str(issue_number)]["status"] == "dispatched"
    relabeled = _events(paths.state_file, "session_failed_relabeled", issue_number)
    assert len(relabeled) == 1
    assert relabeled[0]["payload"]["label_write_ok"] is True
    assert _events(paths.state_file, "local_work_ready", issue_number) == []


def test_orphan_sweep_pr_capable_backend_ignores_local_park(
    tmp_path: Path, shallow_wts: Path
) -> None:
    """Control: the identical git state (worktree with commits ahead) on a
    PR-capable backend keeps the existing reclaim behavior -- the
    capability probe, not the presence of commits, selects the lane."""
    labels_cfg = LabelConfig()
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = OrchestratorConfig(claude_code=ClaudeCodeConfig(worktrees_dir=str(shallow_wts)))

    issue_number = 42
    branch = "agent/issue-42-x"
    _add_worktree_commit(repo_root, config, branch)

    class _RepoRootGitHub(FakeGitHub):
        """FakeGitHub defaults ``publishes_pull_requests`` to True (absent
        attribute); adding ``repo_root`` exercises the gate on identical
        git state rather than on a missing repo_root."""

        def __init__(self, repo: Path) -> None:
            super().__init__()
            self.repo_root = repo

    fake_gh = _RepoRootGitHub(repo_root)
    fake_gh.issues = [
        {
            "number": issue_number,
            "title": "fix the flaky test",
            "url": "https://example.test/issues/42",
            "body": "",
            "labels": [
                {"name": config.labels.in_progress},
                {"name": config.labels.ready},
            ],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    _seed_dead_dispatched(paths.state_file, issue_number, branch)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    _run_sweep(
        sessions_dir,
        paths.state_file,
        config,
        fake_gh,
        _wg(paths.state_file),
        tmp_path / "fleet",
    )

    # Normal reclaim ran; the park lane never touched anything.
    assert (issue_number, labels_cfg.in_progress) in fake_gh.labels_removed
    assert _events(paths.state_file, "local_work_ready", issue_number) == []
    assert len(_events(paths.state_file, "session_failed_relabeled", issue_number)) == 1
    state = load_state(paths.state_file)
    assert state["issues"][str(issue_number)]["status"] == "dispatched"


# ---------------------------------------------------------------------------
# Failure / dry-run paths
# ---------------------------------------------------------------------------


def test_orphan_sweep_park_label_failure_is_loud_and_retryable(
    tmp_path: Path, shallow_wts: Path
) -> None:
    """A failed park label write must NOT silently strand the active label
    or falsely advance the status: the sweep falls through to the normal
    reclaim (the loud outcome -- redispatch cap / backstop still apply) and
    the ``session_failed_relabeled`` event carries the park failure. A
    later pass once the label backend recovers parks successfully."""
    labels_cfg = LabelConfig()
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _local_config(repo_root, shallow_wts)

    issue_number = 55
    branch = "agent/issue-55-x"
    _add_worktree_commit(repo_root, config, branch)

    issues_dir = repo_root / config.local_issues.issues_dir
    _write_issue(
        issues_dir,
        issue_number,
        title="fix the flaky test",
        labels=(labels_cfg.ready, labels_cfg.in_progress),
    )

    class _FlakyLabelLocalGitHub(LocalFileGitHub):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.fail_labels = True

        def add_issue_label(self, number: int, label: str) -> bool:
            if self.fail_labels:
                return False
            return super().add_issue_label(number, label)

        def remove_issue_label(self, number: int, label: str) -> bool:
            if self.fail_labels:
                return False
            return super().remove_issue_label(number, label)

    gh = _FlakyLabelLocalGitHub(repo_root=repo_root, issues_dir=issues_dir)

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    _seed_dead_dispatched(paths.state_file, issue_number, branch)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: label backend down. Park fails, reclaim also fails its label
    # writes, but the event must record WHY commits were not parked.
    _run_sweep(
        sessions_dir, paths.state_file, config, gh, _wg(paths.state_file), tmp_path / "fleet"
    )

    state = load_state(paths.state_file)
    entry = state["issues"][str(issue_number)]
    # Park did not falsely advance the entry; active label still in place
    # so the issue remains recoverable on a later pass.
    assert entry["status"] == "dispatched"
    assert labels_cfg.in_progress in _label_names(gh, issue_number)
    assert labels_cfg.review_ready not in _label_names(gh, issue_number)

    relabeled = _events(paths.state_file, "session_failed_relabeled", issue_number)
    assert len(relabeled) == 1
    payload = relabeled[0]["payload"]
    assert payload["salvage_failed"] is True
    assert "label transition failed" in payload["salvage_error"]
    assert payload["label_write_ok"] is False

    # Pass 2: backend recovered -- the SAME sweep parks the work.
    gh.fail_labels = False
    _run_sweep(
        sessions_dir, paths.state_file, config, gh, _wg(paths.state_file), tmp_path / "fleet"
    )

    current = _label_names(gh, issue_number)
    assert labels_cfg.review_ready in current
    assert labels_cfg.in_progress not in current
    state = load_state(paths.state_file)
    assert state["issues"][str(issue_number)]["status"] == "open_passive"
    # One local_work_ready event per park attempt: pass 1's failure record
    # (label_write_ok=False), then pass 2's success.
    park_events = _events(paths.state_file, "local_work_ready", issue_number)
    assert [e["payload"]["label_write_ok"] for e in park_events] == [False, True]


def test_orphan_sweep_park_dry_run_writes_nothing(tmp_path: Path, shallow_wts: Path) -> None:
    """Dry-run: the park's label transition and the status flip are both
    WriteGate-gated, so a dry-run sweep must leave the issue file and the
    state file byte-identical while still reporting the issue as handled
    (the sweep ``continue``s on the dry-run ``label_ok``)."""
    labels_cfg = LabelConfig()
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _local_config(repo_root, shallow_wts)

    issue_number = 66
    branch = "agent/issue-66-x"
    _add_worktree_commit(repo_root, config, branch)

    issues_dir = repo_root / config.local_issues.issues_dir
    issue_path = _write_issue(
        issues_dir,
        issue_number,
        title="fix the flaky test",
        labels=(labels_cfg.ready, labels_cfg.in_progress),
    )
    before_issue = issue_path.read_bytes()

    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir, dry_run=True)

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    _seed_dead_dispatched(paths.state_file, issue_number, branch)
    before_state = paths.state_file.read_bytes()

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    _run_sweep(
        sessions_dir,
        paths.state_file,
        config,
        gh,
        _wg(paths.state_file, dry_run=True),
        tmp_path / "fleet",
    )

    assert issue_path.read_bytes() == before_issue
    assert paths.state_file.read_bytes() == before_state
    state = load_state(paths.state_file)
    assert state["issues"][str(issue_number)]["status"] == "dispatched"
