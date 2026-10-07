"""Issue #1971 regression: the ``dead_dispatched_reap_minutes`` timer
backstop must not escalate a dead worker's committed work on a no-PR
backend.

``orphaned_worker_sweep.maybe_reap_dead_dispatched_worker`` is a pure
timer: once ``orphan_drift_at`` has been armed longer than the reap window
it escalates the still-``dispatched`` entry -- on a ``local_issues``
backend without ever checking whether the worker's branch carries commits.
Issue #1923 parked salvageable work only for orphans that still carried an
*active* label, so the residual class this file pins is the labelless one:
a dispatched entry whose label is already gone (failed label write, prior
partial pass) still had commits on its branch, was never probed, and was
escalated by the backstop as if the worker had failed.

The fix park-probes every backstop-due no-PR orphan BEFORE the sweep's
state lock (``park_unpublishable_work`` takes ``state_lock`` itself, so it
can never run inside the locked classification) and threads the verdict
in: a parked issue leaves ``dispatched`` and is skipped by the in-lock
status re-verify; a failed park or inconclusive branch probe lands in
``local_park_deferred`` and defers the escalation one pass.

Covered here:

- labelless backstop-due orphan with commits -> parked ``review_ready`` /
  ``open_passive``, ``local_work_ready`` event, no reap escalation;
- branch-diff probe failure (branch ref exists, diff errors) -> deferred,
  ``orphan_drift_at`` still armed, fingerprinted ``orphaned_worker_drift``
  audit event, and the next pass parks;
- park label-write failure on the same shape -> deferred, not escalated,
  then parks on retry;
- labelless orphan whose branch ref provably does not exist -> the #654
  escalation still fires (control: deferral only when salvage is
  unproven);
- identical git/label state on a PR-capable backend -> unchanged
  escalation (the capability gate, not the git state, selects the lane);
- the session-file reap lane (``dead_worker_reap``) parks a labelless
  ``dispatched`` session's branch instead of skipping it.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import charlie_work.dead_worker_sweep.effects_pr as effects_pr
from _dead_session_fixtures import _write_dead_session_sidecar
from _fakes_github import FakeGitHub
from _local_park_fixtures import (
    _add_worktree_commit,
    _events,
    _git,
    _init_repo,
    _label_names,
    _local_config,
    _run_sweep,
    _seed_dead_dispatched,
    _sessions_dir,
    _wg,
    _write_issue,
)
from _local_park_fixtures import _shallow_wts as _register_shallow_wts  # noqa: F401
from charlie_work.config import (
    ClaudeCodeConfig,
    LabelConfig,
    OrchestratorConfig,
)
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state

# ---------------------------------------------------------------------------
# The #1971 residual: a backstop-due orphan with no active label still has
# its committed branch probed and parked instead of escalated.
# ---------------------------------------------------------------------------


def test_backstop_due_labelless_local_orphan_parks(
    tmp_path: Path, shallow_wts: Path, monkeypatch
) -> None:
    """The issue #1971 shape: the worker committed locally and exited, the
    active label is already gone (the #1923 reclaim lane skips such issues),
    and ``orphan_drift_at`` has long expired. Before the fix the timer
    backstop escalated this entry without ever looking at the branch --
    the deliverable. Now the pre-lock park probe runs first."""
    labels_cfg = LabelConfig()
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _local_config(repo_root, shallow_wts)

    issue_number = 1971
    branch = "agent/issue-1971-x"
    _add_worktree_commit(repo_root, config, branch)

    issues_dir = repo_root / config.local_issues.issues_dir
    # No active label -- the residual class #1923 left uncovered.
    _write_issue(issues_dir, issue_number, title="fix the flaky test", labels=())
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    _seed_dead_dispatched(paths.state_file, issue_number, branch)

    _run_sweep(
        _sessions_dir(tmp_path),
        paths.state_file,
        config,
        gh,
        _wg(paths.state_file),
        tmp_path / "fleet",
        monkeypatch=monkeypatch,
    )

    current = _label_names(gh, issue_number)
    assert labels_cfg.review_ready in current
    assert labels_cfg.operator_queue not in current
    assert labels_cfg.human_needed not in current

    state = load_state(paths.state_file)
    entry = state["issues"][str(issue_number)]
    assert entry["status"] == "open_passive"
    assert "orphan_drift_at" not in entry

    assert len(_events(paths.state_file, "local_work_ready", issue_number)) == 1
    assert _events(paths.state_file, "dead_dispatched_worker_reaped", issue_number) == []


def test_backstop_due_probe_failure_defers_escalation(
    tmp_path: Path, shallow_wts: Path, monkeypatch
) -> None:
    """An inconclusive branch probe must NEVER read as no-work: with the
    worktree gone the park lane falls back to ``branch_diff``; when the diff
    errors while the branch ref still exists the verdict is
    ``probe_failed`` -- the in-lock backstop defers one pass instead of
    escalating over possibly-salvageable commits, and the next pass (probe
    healthy again) parks."""
    labels_cfg = LabelConfig()
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _local_config(repo_root, shallow_wts)

    issue_number = 1972
    branch = "agent/issue-1972-x"
    worktree_path = _add_worktree_commit(repo_root, config, branch)
    # Worktree gone, branch kept -- the branch-ref fallback is what probes.
    _git(repo_root, "worktree", "remove", "--force", str(worktree_path))

    issues_dir = repo_root / config.local_issues.issues_dir
    _write_issue(issues_dir, issue_number, title="fix the flaky test", labels=())
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    _seed_dead_dispatched(paths.state_file, issue_number, branch)

    sessions_dir = _sessions_dir(tmp_path)

    # Pass 1: the branch-diff probe errors (transient git failure) while the
    # ref provably exists -> probe_failed -> deferred, not escalated.
    with patch(
        "charlie_work.local_work_park.branch_diff_result",
        return_value=(None, "git diff main...b: fatal: bad revision"),
    ):
        _run_sweep(
            sessions_dir,
            paths.state_file,
            config,
            gh,
            _wg(paths.state_file),
            tmp_path / "fleet",
            monkeypatch=monkeypatch,
        )

    state = load_state(paths.state_file)
    entry = state["issues"][str(issue_number)]
    assert entry["status"] == "dispatched"
    # The drift marker stays armed so the next pass re-probes.
    assert entry.get("orphan_drift_at") is not None
    assert _events(paths.state_file, "dead_dispatched_worker_reaped", issue_number) == []
    assert labels_cfg.review_ready not in _label_names(gh, issue_number)
    drift = _events(paths.state_file, "orphaned_worker_drift", issue_number)
    assert len(drift) == 1
    assert drift[0]["payload"]["reason"] == "dead_dispatched_local_park_deferred"

    # Pass 2: probe healthy again -> the real diff is non-empty -> parked.
    _run_sweep(
        sessions_dir,
        paths.state_file,
        config,
        gh,
        _wg(paths.state_file),
        tmp_path / "fleet",
        monkeypatch=monkeypatch,
    )
    state = load_state(paths.state_file)
    assert state["issues"][str(issue_number)]["status"] == "open_passive"
    assert labels_cfg.review_ready in _label_names(gh, issue_number)
    assert _events(paths.state_file, "dead_dispatched_worker_reaped", issue_number) == []


def test_backstop_due_park_label_failure_defers_escalation(
    tmp_path: Path, shallow_wts: Path, monkeypatch
) -> None:
    """A failed ``local_work_ready`` label write must not escalate in the
    same pass: the issue may still carry salvageable commits, so the
    backstop defers and the next pass (label backend healthy) parks."""
    labels_cfg = LabelConfig()
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _local_config(repo_root, shallow_wts)

    issue_number = 1973
    branch = "agent/issue-1973-x"
    _add_worktree_commit(repo_root, config, branch)

    issues_dir = repo_root / config.local_issues.issues_dir
    _write_issue(issues_dir, issue_number, title="fix the flaky test", labels=())

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

    sessions_dir = _sessions_dir(tmp_path)

    _run_sweep(
        sessions_dir,
        paths.state_file,
        config,
        gh,
        _wg(paths.state_file),
        tmp_path / "fleet",
        monkeypatch=monkeypatch,
    )

    state = load_state(paths.state_file)
    entry = state["issues"][str(issue_number)]
    assert entry["status"] == "dispatched"
    assert entry.get("orphan_drift_at") is not None
    assert _events(paths.state_file, "dead_dispatched_worker_reaped", issue_number) == []
    assert labels_cfg.review_ready not in _label_names(gh, issue_number)
    assert labels_cfg.operator_queue not in _label_names(gh, issue_number)

    # Pass 2: label backend recovered -> the same branch parks.
    gh.fail_labels = False
    _run_sweep(
        sessions_dir,
        paths.state_file,
        config,
        gh,
        _wg(paths.state_file),
        tmp_path / "fleet",
        monkeypatch=monkeypatch,
    )
    state = load_state(paths.state_file)
    assert state["issues"][str(issue_number)]["status"] == "open_passive"
    assert labels_cfg.review_ready in _label_names(gh, issue_number)


def test_backstop_due_no_branch_ref_still_escalates(
    tmp_path: Path, shallow_wts: Path, monkeypatch
) -> None:
    """Control: a labelless backstop-due orphan whose branch ref provably
    does not exist gets the unchanged #654 escalation -- deferral only
    applies while salvage is unproven."""
    labels_cfg = LabelConfig()
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _local_config(repo_root, shallow_wts)

    issue_number = 1974
    branch = "agent/issue-1974-x"
    # No worktree, no branch -- the ref probe proves there is no work.

    issues_dir = repo_root / config.local_issues.issues_dir
    _write_issue(issues_dir, issue_number, title="fix the flaky test", labels=())
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    _seed_dead_dispatched(paths.state_file, issue_number, branch)

    _run_sweep(
        _sessions_dir(tmp_path),
        paths.state_file,
        config,
        gh,
        _wg(paths.state_file),
        tmp_path / "fleet",
        monkeypatch=monkeypatch,
    )

    state = load_state(paths.state_file)
    entry = state["issues"][str(issue_number)]
    assert entry["status"] == "escalated"
    assert entry.get("escalation_reason") == "dead_dispatched_worker_reap"
    assert labels_cfg.operator_queue in _label_names(gh, issue_number)
    assert len(_events(paths.state_file, "dead_dispatched_worker_reaped", issue_number)) == 1
    assert _events(paths.state_file, "local_work_ready", issue_number) == []


def test_backstop_due_pr_backend_still_escalates(
    tmp_path: Path, shallow_wts: Path, monkeypatch
) -> None:
    """Control: identical git/label state on a PR-capable backend keeps the
    unchanged #654 escalation -- the capability probe, not the branch's
    commits, selects the lane."""
    labels_cfg = LabelConfig()
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = OrchestratorConfig(claude_code=ClaudeCodeConfig(worktrees_dir=str(shallow_wts)))

    issue_number = 1975
    branch = "agent/issue-1975-x"
    _add_worktree_commit(repo_root, config, branch)

    class _RepoRootGitHub(FakeGitHub):
        """Adds ``repo_root`` so the drain is skipped on the capability
        flag alone, not on a missing repo root."""

        def __init__(self, repo: Path) -> None:
            super().__init__()
            self.repo_root = repo

    fake_gh = _RepoRootGitHub(repo_root)
    fake_gh.issues = [
        {
            "number": issue_number,
            "title": "fix the flaky test",
            "url": "https://example.test/issues/1975",
            "body": "",
            "labels": [],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    _seed_dead_dispatched(paths.state_file, issue_number, branch)

    _run_sweep(
        _sessions_dir(tmp_path),
        paths.state_file,
        config,
        fake_gh,
        _wg(paths.state_file),
        tmp_path / "fleet",
        monkeypatch=monkeypatch,
    )

    state = load_state(paths.state_file)
    entry = state["issues"][str(issue_number)]
    assert entry["status"] == "escalated"
    assert entry.get("escalation_reason") == "dead_dispatched_worker_reap"
    assert (issue_number, labels_cfg.operator_queue) in fake_gh.labels_added
    assert _events(paths.state_file, "local_work_ready", issue_number) == []


# ---------------------------------------------------------------------------
# The session-file reap lane: ``if not active_labels`` must still park the
# committed branch on a no-PR backend rather than skip the issue entirely.
# ---------------------------------------------------------------------------


def test_dead_session_reap_parks_labelless_local_work(tmp_path: Path, shallow_wts: Path) -> None:
    """Issue #1971 path 3: the sidecar-based dead-session reap's
    ``if not active_labels: continue`` gate is right on a PR backend, but on
    a no-PR backend it dropped labelless committed work to the timer
    backstop. Here the state entry still reads ``dispatched`` and carries
    no ``branch_name`` -- the sidecar's recorded branch backfills it."""
    labels_cfg = LabelConfig()
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _local_config(repo_root, shallow_wts)

    issue_number = 1976
    branch = "agent/issue-1976-x"
    worktree_path = _add_worktree_commit(repo_root, config, branch)

    issues_dir = repo_root / config.local_issues.issues_dir
    _write_issue(issues_dir, issue_number, title="fix the flaky test", labels=(labels_cfg.ready,))
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                # Deliberately no branch_name: the sidecar's ``branch``
                # field is what the park lane must use.
                "issues": {str(issue_number): {"status": "dispatched"}},
                "prs": {},
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    _write_dead_session_sidecar(sessions_dir, issue_number, branch, worktree_path)

    from charlie_work.workflow import (
        _classify_dead_sessions_and_update_throttle_state,
    )

    with patch.object(effects_pr, "push_branch") as push_mock:
        push_mock.side_effect = AssertionError(
            "push_branch must not be called for a no-PR backend"
        )
        _classify_dead_sessions_and_update_throttle_state(
            sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
        )

    current = _label_names(gh, issue_number)
    assert labels_cfg.review_ready in current
    assert labels_cfg.operator_queue not in current
    state = load_state(state_file)
    assert state["issues"][str(issue_number)]["status"] == "open_passive"
    assert len(_events(state_file, "local_work_ready", issue_number)) == 1


def test_dead_session_reap_terminal_label_not_reparked(tmp_path: Path, shallow_wts: Path) -> None:
    """Control: a dead labelless-active session whose issue already carries
    a terminal label (``agent:done``) keeps it -- the ``local_work_ready``
    edge strips every workflow label, so parking here would overwrite an
    answer another lane already gave."""
    labels_cfg = LabelConfig()
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _local_config(repo_root, shallow_wts)

    issue_number = 1977
    branch = "agent/issue-1977-x"
    worktree_path = _add_worktree_commit(repo_root, config, branch)

    issues_dir = repo_root / config.local_issues.issues_dir
    _write_issue(
        issues_dir,
        issue_number,
        title="fix the flaky test",
        labels=(labels_cfg.done,),
    )
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "issues": {str(issue_number): {"status": "dispatched"}},
                "prs": {},
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    _write_dead_session_sidecar(sessions_dir, issue_number, branch, worktree_path)

    from charlie_work.workflow import (
        _classify_dead_sessions_and_update_throttle_state,
    )

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
    )

    current = _label_names(gh, issue_number)
    assert labels_cfg.done in current
    assert labels_cfg.review_ready not in current
    assert _events(state_file, "local_work_ready", issue_number) == []
