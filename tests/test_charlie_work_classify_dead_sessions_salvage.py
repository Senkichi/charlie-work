"""Dead-session classification salvage and API-session settlement: unpublished-work salvage, dirty-worktree salvage, salvage skip conditions, salvage-push fallback, and api-session budget-ledger settlement / provider-auth classification.

Split out of ``tests/test_charlie_work.py`` (issue #1551, Track-1
wave 5/8).
"""

from __future__ import annotations

import json
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from unittest.mock import patch
import pytest
from _dead_session_fixtures import (
    _git,
    _write_dead_session_sidecar,
    _make_classify_state,
)
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg
from _worktree_fixtures import (
    _init_bare_remote_and_clone,
    _setup_completed_worktree,
)
from charlie_work import github as github_module
from charlie_work.config import (
    OrchestratorConfig,
    PostMortemConfig,
)
from charlie_work.state import load_state


def test_classify_dead_sessions_salvages_completed_unpublished_work(
    tmp_path: Path,
) -> None:
    """Issue #252: a clean, ahead worktree is salvaged (push + PR + pr_open label)."""
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 252)
    sessions_dir, state_file = _make_classify_state(tmp_path)
    _write_dead_session_sidecar(sessions_dir, 252, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    gh.issues = [
        {
            "number": 252,
            "title": "Test issue",
            "url": "https://example.test/issues/252",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.pr_create_return = 101

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
    )

    # Branch pushed and PR created
    remote_refs = _git(remote, "show-ref")
    assert "agent/issue-252" in remote_refs.stdout
    assert len(gh.prs_created) == 1
    assert gh.prs_created[0]["head"] == branch
    assert gh.prs_created[0]["base"] == "main"

    # Labels moved to pr_open
    assert (252, config.labels.in_progress) in gh.labels_removed
    assert (252, config.labels.pr_open) in gh.labels_added

    # Sidecar reaped and event recorded
    assert not (sessions_dir / "issue-252.json").exists()
    state = json.loads(state_file.read_text(encoding="utf-8"))
    events = [e for e in state["events"] if e["kind"] == "session_salvaged"]
    assert len(events) == 1
    assert events[0]["payload"]["issue_number"] == 252
    assert events[0]["payload"]["pr_number"] == 101


def test_classify_dead_sessions_dirty_worktree_with_commits_salvaged(tmp_path: Path) -> None:
    """Issue #1130: a dirty worktree that has commits ahead of base IS salvaged.
    The committed work is pushed and a PR is opened; the working-tree dirt
    (e.g. shim/scaffolding artifacts not in ``injected_paths``) is irrelevant
    to the push and survives in the worktree for later inspection."""
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 253, dirty=True)
    sessions_dir, state_file = _make_classify_state(tmp_path)
    _write_dead_session_sidecar(sessions_dir, 253, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    gh.issues = [
        {
            "number": 253,
            "title": "Test issue",
            "url": "https://example.test/issues/253",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.pr_create_return = 101

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
    )

    # Branch pushed and PR created — the committed work is salvaged.
    remote_refs = _git(remote, "show-ref")
    assert branch in remote_refs.stdout
    assert len(gh.prs_created) == 1
    assert gh.prs_created[0]["head"] == branch
    # Labels moved to pr_open, not ready.
    assert (253, config.labels.in_progress) in gh.labels_removed
    assert (253, config.labels.pr_open) in gh.labels_added
    assert (253, config.labels.ready) not in gh.labels_added

    state = json.loads(state_file.read_text(encoding="utf-8"))
    events = [e for e in state["events"] if e["kind"] == "session_salvaged"]
    assert len(events) == 1
    assert events[0]["payload"]["issue_number"] == 253


def test_classify_dead_sessions_skips_salvage_when_issue_closed(tmp_path: Path) -> None:
    """Issue #1221 (check 1): salvage refuses to open a PR when the linked
    issue is already CLOSED. The dead session's snapshot is stale -- an
    operator/sibling merged the work and closed the issue inside the staleness
    window -- so salvage re-checks live issue state at fire time and downgrades
    to a ``salvage_skipped_already_landed`` event instead of a vestigial PR.
    """
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 1221)
    sessions_dir, state_file = _make_classify_state(tmp_path)
    _write_dead_session_sidecar(sessions_dir, 1221, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    # Closed issue still carrying an active label (the secondary defect in
    # #1221): the active-label gate lets the lane proceed, and salvage's own
    # closed-issue check must refuse the PR.
    gh.issues = [
        {
            "number": 1221,
            "title": "Salvage race",
            "url": "https://example.test/issues/1221",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "CLOSED",
        }
    ]
    gh.pr_create_return = 999  # would-be vestigial salvage PR

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
    )

    # No vestigial PR opened.
    assert not gh.prs_created
    # Active label is NOT stripped by the skip path (label cleanup is the
    # reconcile lane's job); the issue is not re-dispatched.
    assert (1221, config.labels.in_progress) not in gh.labels_removed
    assert (1221, config.labels.ready) not in gh.labels_added

    state = json.loads(state_file.read_text(encoding="utf-8"))
    skip_events = [e for e in state["events"] if e["kind"] == "salvage_skipped_already_landed"]
    assert len(skip_events) == 1
    assert skip_events[0]["payload"]["issue_number"] == 1221
    assert skip_events[0]["payload"]["reason"] == "issue_closed"
    # The skip path does NOT remove labels -- the payload records active_labels
    # (what the issue carried at skip time), not removed_labels.
    assert "removed_labels" not in skip_events[0]["payload"]
    assert skip_events[0]["payload"]["active_labels"] == [config.labels.in_progress]
    # No session_salvaged event was emitted.
    assert not [e for e in state["events"] if e["kind"] == "session_salvaged"]


def test_classify_dead_sessions_skips_salvage_when_pr_merged(tmp_path: Path) -> None:
    """Issue #1221 (check 2): salvage refuses to open a PR when a PR binding to
    the issue is already MERGED, even if the GitHub issue is still OPEN (the
    close event lags or the merge closed it after the snapshot)."""
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 1221)
    sessions_dir, state_file = _make_classify_state(tmp_path)
    _write_dead_session_sidecar(sessions_dir, 1221, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    gh.issues = [
        {
            "number": 1221,
            "title": "Salvage race",
            "url": "https://example.test/issues/1221",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    # A merged PR whose head ref binds to issue 1221 via the branch prefix.
    gh.prs = [
        {
            "number": 1217,
            "title": "Fix #1221",
            "url": "https://example.test/pull/1217",
            "headRefName": branch,
            "baseRefName": "main",
            "headRefOid": "sha-merged",
            "body": "Closes #1221",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        }
    ]
    gh.pr_create_return = 999

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
    )

    assert not gh.prs_created
    state = json.loads(state_file.read_text(encoding="utf-8"))
    skip_events = [e for e in state["events"] if e["kind"] == "salvage_skipped_already_landed"]
    assert len(skip_events) == 1
    assert skip_events[0]["payload"]["issue_number"] == 1221
    assert skip_events[0]["payload"]["reason"] == "pr_merged"
    assert "removed_labels" not in skip_events[0]["payload"]
    assert skip_events[0]["payload"]["active_labels"] == [config.labels.in_progress]
    assert not [e for e in state["events"] if e["kind"] == "session_salvaged"]


def test_classify_dead_sessions_skips_salvage_when_branch_empty_diff(
    tmp_path: Path,
) -> None:
    """Issue #1221 (check 3): salvage refuses to open a PR when the branch's
    tree is identical to current main's tree -- the work already landed, so a
    salvage PR would be vestigial.

    Reproduces the exact race: ``inspect_worktree_state`` resolves its base
    against a *stale* ``origin/main`` tracking ref (so it sees COMPLETED, ahead
    of the old tip), while the live remote main has already advanced to include
    the branch's work. Salvage fetches the live tip before opening a PR and
    detects the empty tree diff.
    """
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 1221)
    sessions_dir, state_file = _make_classify_state(tmp_path)
    _write_dead_session_sidecar(sessions_dir, 1221, branch, worktree_path)

    # Push the worker branch so a second clone can merge it into main.
    _git(repo_root, "push", "origin", branch)

    # Advance origin/main to include the branch's work via a SECOND clone,
    # leaving repo_root's origin/main tracking ref stale -- the race window
    # between the merge landing and the dead session's staleness tripping.
    clone2 = tmp_path / "clone2"
    clone2.mkdir(parents=True, exist_ok=True)
    _git(clone2, "init", "--initial-branch=main")
    _git(clone2, "config", "user.email", "test@example.test")
    _git(clone2, "config", "user.name", "Test User")
    _git(clone2, "config", "commit.gpgSign", "false")
    _git(clone2, "remote", "add", "origin", str(remote))
    _git(clone2, "fetch", "origin")
    _git(clone2, "merge", "--ff-only", f"origin/{branch}")
    _git(clone2, "push", "origin", "main")

    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    gh.issues = [
        {
            "number": 1221,
            "title": "Salvage race",
            "url": "https://example.test/issues/1221",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.pr_create_return = 999

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
    )

    # No vestigial PR: the branch contributes nothing beyond current main.
    assert not gh.prs_created
    state = json.loads(state_file.read_text(encoding="utf-8"))
    skip_events = [e for e in state["events"] if e["kind"] == "salvage_skipped_already_landed"]
    assert len(skip_events) == 1
    assert skip_events[0]["payload"]["issue_number"] == 1221
    assert skip_events[0]["payload"]["reason"] == "empty_diff"
    assert "removed_labels" not in skip_events[0]["payload"]
    assert skip_events[0]["payload"]["active_labels"] == [config.labels.in_progress]
    assert not [e for e in state["events"] if e["kind"] == "session_salvaged"]


def test_classify_dead_sessions_salvage_proceeds_when_merged_pr_search_fails(
    tmp_path: Path,
) -> None:
    """Issue #1221 (check 2 fail-safe): when ``merged_prs_for_issue`` returns
    ``ok=False``, salvage must NOT treat that as evidence of a merge. It falls
    through to the empty-diff check (check 3) and, if the branch has real work,
    opens a PR -- a human reviews salvage PRs anyway.

    The result carries a non-empty list with ``ok=False`` so the test exercises
    the ``ok`` flag specifically: without it, ``len(merged) > 0`` alone would
    trigger ``pr_merged`` and suppress the PR.
    """
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 1221)
    sessions_dir, state_file = _make_classify_state(tmp_path)
    _write_dead_session_sidecar(sessions_dir, 1221, branch, worktree_path)

    config = OrchestratorConfig()

    class FakeGitHubFailingMergeSearch(FakeGitHub):
        def merged_prs_for_issue(self, issue_number: int, branch_prefix: str):
            # Non-empty list with ok=False: a failed search that happened to
            # return items must NOT be treated as a merge.
            return github_module._MergedPRSearchResult(
                [{"number": 1217, "state": "MERGED"}], ok=False
            )

    gh = FakeGitHubFailingMergeSearch(repo_root=repo_root)
    gh.issues = [
        {
            "number": 1221,
            "title": "Salvage race",
            "url": "https://example.test/issues/1221",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.pr_create_return = 888

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
    )

    # Salvage proceeded: the ok=False search fell through to the empty-diff
    # check, which also did not fire (branch has real work), so a PR was opened.
    assert len(gh.prs_created) == 1
    assert gh.prs_created[0]["head"] == branch
    # No skip event -- the fail-safe worked.
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert not [e for e in state["events"] if e["kind"] == "salvage_skipped_already_landed"]
    salvaged = [e for e in state["events"] if e["kind"] == "session_salvaged"]
    assert len(salvaged) == 1
    assert salvaged[0]["payload"]["pr_number"] == 888


def test_classify_dead_sessions_salvage_push_failure_fallback(tmp_path: Path) -> None:
    """Issue #252: a failed salvage push records failure and falls back to relabel."""
    # Issue #1317: push_branch is called bare-name from inside _attempt_salvage,
    # which moved (verbatim) to dead_worker_reap.py -- so the bare-name lookup
    # now resolves via dead_worker_reap.py's own globals, not workflow.py's.
    # Patch it there, not on the (still-valid) workflow.py facade re-export of
    # _classify_dead_sessions_and_update_throttle_state itself.
    from charlie_work import dead_worker_reap as workflow_module
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 255)
    sessions_dir, state_file = _make_classify_state(tmp_path)
    _write_dead_session_sidecar(sessions_dir, 255, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    gh.issues = [
        {
            "number": 255,
            "title": "Test issue",
            "url": "https://example.test/issues/255",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.pr_create_return = 101

    original_push_branch = workflow_module.push_branch
    workflow_module.push_branch = lambda repo, br, worktree_path=None, **kw: (
        False,
        "simulated push failure",
    )
    try:
        _classify_dead_sessions_and_update_throttle_state(
            sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
        )
    finally:
        workflow_module.push_branch = original_push_branch

    # No PR created, active label removed, ready label added, event records salvage failure
    assert not gh.prs_created
    assert (255, config.labels.in_progress) in gh.labels_removed
    assert (255, config.labels.ready) in gh.labels_added

    state = json.loads(state_file.read_text(encoding="utf-8"))
    events = [e for e in state["events"] if e["kind"] == "session_failed_relabeled"]
    assert len(events) == 1
    assert events[0]["payload"].get("salvage_failed") is True


def test_classify_dead_sessions_dead_api_session_settles_budget_ledger(
    tmp_path: Path,
) -> None:
    """Confirmed-dead api-worker session: the dead lane reaps and settles spend.

    Covers the dead-session reap call site (~workflow.py:2652). A wiring
    regression that drops ``api_config``/``state_dir`` from that call leaves
    the sidecar reaped but the ledger empty — this assertion fails.
    """
    from _api_budget_fixtures import (
        api_worker_config,
        ledger_entries,
        write_api_events,
        write_api_sidecar,
    )
    from charlie_work.config import PostMortemConfig
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig(
        api_worker=api_worker_config(),
        post_mortem=PostMortemConfig(enabled=False),
    )
    sessions_dir, state_file = _make_classify_state(tmp_path)
    write_api_sidecar(sessions_dir, 42, provider="example")
    write_api_events(sessions_dir, 42)

    gh = FakeGitHub()
    gh.issues = [
        {
            "number": 42,
            "title": "Test issue",
            "url": "https://example.test/issues/42",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.prs = []

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
    )

    sessions = ledger_entries(state_file.parent)
    assert len(sessions) == 1, "dead api session must settle into the ledger"
    entry = sessions[0]
    assert entry.issue == 42
    assert entry.provider == "example"
    assert entry.model == "example-model"
    # 1M*3 + 0.2M*15 + 0.5M*0.30 = 6.15
    assert entry.usd == pytest.approx(6.15)


def test_classify_dead_sessions_launch_failed_api_session_settles_budget_ledger(
    tmp_path: Path,
) -> None:
    """Launch-failed api-worker sidecar: the launch-failure lane settles spend.

    Covers the launch-failure reap call site (~workflow.py:2497), which fires
    for a sidecar with ``pid is None`` and a non-null ``error``. A wiring
    regression that drops the api kwargs from that call leaves the ledger
    empty — this assertion fails.
    """
    from _api_budget_fixtures import (
        api_worker_config,
        ledger_entries,
        write_api_events,
        write_api_sidecar,
    )
    from charlie_work.config import PostMortemConfig
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig(
        api_worker=api_worker_config(),
        post_mortem=PostMortemConfig(enabled=False),
    )
    sessions_dir, state_file = _make_classify_state(tmp_path)
    write_api_sidecar(sessions_dir, 43, provider="example", error="launch failed", pid=None)
    write_api_events(sessions_dir, 43)

    gh = FakeGitHub()
    gh.issues = [
        {
            "number": 43,
            "title": "Test issue",
            "url": "https://example.test/issues/43",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.prs = []

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
    )

    sessions = ledger_entries(state_file.parent)
    assert len(sessions) == 1, "launch-failed api session must settle into the ledger"
    entry = sessions[0]
    assert entry.issue == 43
    assert entry.provider == "example"
    assert entry.usd == pytest.approx(6.15)


def test_classify_dead_sessions_api_provider_auth_classification(
    tmp_path: Path,
) -> None:
    """Issue #484 review finding: the ``elif w.adapter_kind == "api"`` branch in
    ``_classify_dead_sessions_and_update_throttle_state`` (workflow.py
    dead-session lane) classifies a dead api worker with a 401 log tail as
    ``provider_auth`` with a 24h cooldown. A wiring regression that drops this
    branch leaves the failure_kind as the fallback and ``throttled_until``
    unset. The sidecar is reaped by this lane, so the classification is
    asserted via the function's returned reaped-entry and the persisted
    ``throttled_until``.
    """
    from _api_budget_fixtures import api_worker_config, write_api_sidecar
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig(
        api_worker=api_worker_config(),
        post_mortem=PostMortemConfig(enabled=False),
    )
    sessions_dir, state_file = _make_classify_state(tmp_path)
    write_api_sidecar(sessions_dir, 4803, provider="example", pid=99997)

    # 401 log tail — the dead-session lane classifies it.
    log_path = sessions_dir / "issue-4803.claude.log"
    log_path.write_text("Error: 403 Forbidden. Authentication failed.\n", encoding="utf-8")

    gh = FakeGitHub()
    gh.issues = [
        {
            "number": 4803,
            "title": "Test issue",
            "url": "https://example.test/issues/4803",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.prs = []

    before = datetime.now(UTC)
    with patch("charlie_work.worker.is_worker_alive", return_value=False):
        reaped = _classify_dead_sessions_and_update_throttle_state(
            sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
        )

    # The reaped entry carries the resolved failure_kind (provider_auth, not
    # the fallback "stalled"/"unpublished_work").
    api_reaped = [r for r in reaped if r["issue_number"] == 4803]
    assert len(api_reaped) == 1
    assert api_reaped[0]["failure_kind"] == "provider_auth"
    assert api_reaped[0]["adapter_kind"] == "api"

    # 24h cooldown persisted to state.json.
    state = load_state(state_file)
    throttled_until = state.get("throttled_until")
    assert throttled_until is not None
    throttle_time = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    assert before + timedelta(hours=23) <= throttle_time <= before + timedelta(hours=25)


def test_classify_dead_sessions_launch_failed_api_provider_auth(
    tmp_path: Path,
) -> None:
    """Issue #484 review finding: the ``elif w.adapter_kind == "api"`` branch in
    the launch-failure lane of ``_classify_dead_sessions_and_update_throttle_state``
    (workflow.py) classifies a launch-failed api sidecar (pid=None, error set)
    whose log tail contains a 401 signature as ``provider_auth``. A wiring
    regression that drops this branch leaves the launch failure classified as
    the generic ``launch_failed`` fallback. The sidecar is reaped by this lane,
    so the classification is asserted via the returned reaped-entry.
    """
    from _api_budget_fixtures import api_worker_config, write_api_sidecar
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig(
        api_worker=api_worker_config(),
        post_mortem=PostMortemConfig(enabled=False),
    )
    sessions_dir, state_file = _make_classify_state(tmp_path)
    # pid=None + error set → launch-failure lane.
    write_api_sidecar(sessions_dir, 4804, provider="example", pid=None, error="launch failed")

    # 401 log tail — the launch-failure lane classifies it before reaping.
    log_path = sessions_dir / "issue-4804.claude.log"
    log_path.write_text("Error: 401 Unauthorized\n", encoding="utf-8")

    gh = FakeGitHub()
    gh.issues = [
        {
            "number": 4804,
            "title": "Test issue",
            "url": "https://example.test/issues/4804",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.prs = []

    with patch("charlie_work.worker.is_worker_alive", return_value=False):
        reaped = _classify_dead_sessions_and_update_throttle_state(
            sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
        )

    api_reaped = [r for r in reaped if r["issue_number"] == 4804]
    assert len(api_reaped) == 1
    assert api_reaped[0]["failure_kind"] == "provider_auth"
    assert api_reaped[0]["adapter_kind"] == "api"
