"""Regression tests for issue #1241: salvage lane opens duplicate PRs for
already-merged work.

The salvage lane decided purely from worktree shape (dead worker + local
commits) and never checked whether the linked issue was already CLOSED or
whether the commits were already reachable from origin/main. The fix adds a
single enforcement point (`charlie_work.salvage_superseded.check_salvage_superseded`)
called by both salvage lanes (``workflow._attempt_salvage`` and
``reconcile.apply_fixes``'s ``session_unpublished_work_salvaged`` branch), with
a new reachability check (`charlie_work.worktree.salvage_branch_reachable_from_main`)
that catches a merge commit whose tree differs from the salvage head's tree --
the case the #1221 empty-diff check misses.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from _fakes_github import FakeGitHub
from _reconcile_fixtures import (
    _init_bare_remote_and_clone,
    _issue,
    _setup_completed_worktree,
)
from _worktree_fixtures import _git
from charlie_work.config import OrchestratorConfig
from charlie_work.devin_shell import SessionRecord
from charlie_work.instrumentation import read_event_log
from charlie_work.reconcile import DriftItem, apply_fixes
from charlie_work.salvage_superseded import (
    REASON_COMMITS_REACHABLE,
    REASON_ISSUE_CLOSED,
    check_salvage_superseded,
    salvage_skip_event_kind,
)
from charlie_work.instrumentation import _LEVEL_BY_KIND
from charlie_work.state import empty_state
from charlie_work.worktree import salvage_branch_reachable_from_main
from charlie_work.write_gate import WriteGate


def _wg(state_file: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo="charlie-work")


def _make_classify_state(tmp_path: Path) -> tuple[Path, Path]:
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": []}), encoding="utf-8")
    return sessions_dir, state_file


def _write_dead_session_sidecar(
    sessions_dir: Path, issue_number: int, branch: str, worktree_path: Path
) -> None:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    record = SessionRecord(
        issue_number=issue_number,
        branch=branch,
        worktree_path=str(worktree_path),
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(sessions_dir / f"issue-{issue_number}.log"),
        error=None,
    )
    (sessions_dir / f"issue-{issue_number}.json").write_text(
        json.dumps(record.to_dict()), encoding="utf-8"
    )


def _advance_main_past_branch(tmp_path: Path, remote: Path, repo_root: Path, branch: str) -> None:
    """Merge ``branch`` into main via a second clone, then advance main with an
    EXTRA commit so origin/main's tree differs from the branch tip's tree while
    the branch tip remains an ancestor of origin/main -- the exact #1241 shape
    (a merge commit whose tree differs from the salvaged head's tree because
    main moved on with other commits). Leaves repo_root's origin/main tracking
    ref stale so the race window is reproduced.
    """
    _git(repo_root, "push", "origin", branch)
    clone2 = tmp_path / "clone2"
    clone2.mkdir(parents=True, exist_ok=True)
    _git(clone2, "init", "--initial-branch=main")
    _git(clone2, "config", "user.email", "test@example.test")
    _git(clone2, "config", "user.name", "Test User")
    _git(clone2, "config", "commit.gpgSign", "false")
    _git(clone2, "remote", "add", "origin", str(remote))
    _git(clone2, "fetch", "origin")
    _git(clone2, "merge", "--ff-only", f"origin/{branch}")
    # Advance main with an unrelated commit so tree(origin/main) !=
    # tree(branch tip), but the branch tip is still an ancestor.
    (clone2 / "other.txt").write_text("other work\n", encoding="utf-8")
    _git(clone2, "add", "other.txt")
    _git(clone2, "commit", "-m", "unrelated follow-up commit")
    _git(clone2, "push", "origin", "main")


# ---------------------------------------------------------------------------
# salvage_branch_reachable_from_main (the new git reachability helper)
# ---------------------------------------------------------------------------


def test_salvage_branch_reachable_from_main_true_when_ancestor(tmp_path: Path) -> None:
    """Issue #1241: the branch tip is an ancestor of origin/main (its work
    already landed via a merge, then main advanced with an unrelated commit) --
    reachability returns True even though the trees differ (empty-diff would
    miss this)."""
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    _worktree_path, branch = _setup_completed_worktree(repo_root, 1241)
    _advance_main_past_branch(tmp_path, remote, repo_root, branch)

    assert salvage_branch_reachable_from_main(repo_root, branch, "main") is True


def test_salvage_branch_reachable_from_main_false_when_not_ancestor(
    tmp_path: Path,
) -> None:
    """The branch carries a commit that is NOT on origin/main -- reachability
    returns False (salvage should proceed, there is real unlanded work)."""
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    _worktree_path, branch = _setup_completed_worktree(repo_root, 1241)
    # Branch is local-only; origin/main is still the initial commit. The branch
    # tip is not an ancestor of origin/main.
    assert salvage_branch_reachable_from_main(repo_root, branch, "main") is False


def test_salvage_branch_reachable_from_main_false_on_fetch_failure(
    tmp_path: Path,
) -> None:
    """Fail-open: a fetch failure (no origin remote) returns False -- salvage
    proceeds and a human reviews the PR. A duplicate PR is recoverable; silently
    dropped work is not."""
    repo_root = tmp_path / "no-remote-clone"
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init", "--initial-branch=main")
    _git(repo_root, "config", "user.email", "test@example.test")
    _git(repo_root, "config", "user.name", "Test User")
    _git(repo_root, "config", "commit.gpgSign", "false")
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo_root, "add", "README.md")
    _git(repo_root, "commit", "-m", "initial commit")

    assert salvage_branch_reachable_from_main(repo_root, "agent/issue-1241", "main") is False


# ---------------------------------------------------------------------------
# check_salvage_superseded (the shared single enforcement point)
# ---------------------------------------------------------------------------


def test_check_salvage_superseded_commits_reachable(tmp_path: Path) -> None:
    """Issue #1241 core scenario: issue OPEN, no merged PR binds to it (the PR
    search lags), and the branch tip is reachable from origin/main but the
    trees differ (main advanced). The reachability check fires with reason
    ``commits_reachable`` -- the case empty-diff misses."""
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    _worktree_path, branch = _setup_completed_worktree(repo_root, 1241)
    _advance_main_past_branch(tmp_path, remote, repo_root, branch)

    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    gh.issues = [_issue(1241, [config.labels.in_progress], state="OPEN")]
    gh.prs = []
    gh.pr_create_return = 999

    superseded, reason = check_salvage_superseded(
        gh=gh,
        config=config,
        repo_root=repo_root,
        branch=branch,
        base_ref="main",
        issue_number=1241,
        issue=gh.issues[0],
    )
    assert superseded is True
    assert reason == REASON_COMMITS_REACHABLE
    assert salvage_skip_event_kind(reason) == "salvage_skipped_superseded"


def test_check_salvage_superseded_fetches_issue_when_none(tmp_path: Path) -> None:
    """When the caller passes ``issue=None`` (the reconcile lane, which had not
    fetched), the check fetches live state via ``gh.issue_view`` so the
    closed-issue check still fires."""
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    _worktree_path, branch = _setup_completed_worktree(repo_root, 1241)

    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    gh.issues = [_issue(1241, [config.labels.in_progress], state="CLOSED")]
    gh.prs = []

    superseded, reason = check_salvage_superseded(
        gh=gh,
        config=config,
        repo_root=repo_root,
        branch=branch,
        base_ref="main",
        issue_number=1241,
        issue=None,
    )
    assert superseded is True
    assert reason == REASON_ISSUE_CLOSED
    assert salvage_skip_event_kind(reason) == "salvage_skipped_already_landed"


# ---------------------------------------------------------------------------
# workflow.py salvage lane (integration via _classify_dead_sessions)
# ---------------------------------------------------------------------------


def test_classify_dead_sessions_skips_salvage_when_commits_reachable(
    tmp_path: Path,
) -> None:
    """Issue #1241 end-to-end: the dead-worker salvage lane re-checks
    reachability before opening a PR. The branch's work already landed on main
    (via a sibling merge) and main advanced with an unrelated commit, so the
    tree differs (empty-diff would miss it) but the branch tip is an ancestor
    of origin/main. Salvage skips with a ``salvage_skipped_superseded`` event
    and opens NO vestigial PR."""
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 1241)
    _advance_main_past_branch(tmp_path, remote, repo_root, branch)
    sessions_dir, state_file = _make_classify_state(tmp_path)
    _write_dead_session_sidecar(sessions_dir, 1241, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    # Issue still OPEN (the close event lags the merge), and no merged PR binds
    # to it in the fake's snapshot (the PR search lags) -- so only the
    # reachability check catches the already-landed work.
    gh.issues = [_issue(1241, [config.labels.in_progress], state="OPEN")]
    gh.prs = []
    gh.pr_create_return = 999  # would-be vestigial salvage PR

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
    )

    assert not gh.prs_created, "no vestigial salvage PR should be opened"

    state = json.loads(state_file.read_text(encoding="utf-8"))
    superseded_events = [e for e in state["events"] if e["kind"] == "salvage_skipped_superseded"]
    assert len(superseded_events) == 1
    assert superseded_events[0]["payload"]["issue_number"] == 1241
    assert superseded_events[0]["payload"]["reason"] == REASON_COMMITS_REACHABLE
    # No session_salvaged event was emitted.
    assert not [e for e in state["events"] if e["kind"] == "session_salvaged"]


# ---------------------------------------------------------------------------
# reconcile.py salvage lane (integration via apply_fixes)
# ---------------------------------------------------------------------------


def test_reconcile_salvage_skips_when_issue_closed(tmp_path: Path) -> None:
    """Issue #1241: the reconcile salvage lane (``session_unpublished_work_salvaged``)
    previously had NO supersession check and opened a vestigial duplicate PR
    whenever the work had already landed. With the shared check wired in, a
    CLOSED linked issue skips the push/PR, emits ``salvage_skipped_already_landed``,
    and does NOT relabel to ready (the work already landed -- redispatch would
    loop)."""
    from _reconcile_fixtures import FakeGitHub as ReconcileFakeGitHub

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 1241)

    config = OrchestratorConfig()
    gh = ReconcileFakeGitHub(
        prs=[],
        issues=[_issue(1241, [config.labels.in_progress], state="CLOSED")],
        repo_root=repo_root,
        pr_create_return=999,
    )

    drift = [
        DriftItem(
            kind="session_unpublished_work_salvaged",
            issue_number=1241,
            pr_number=None,
            detail="salvage",
            fix_actions=("push", "pr_create"),
            remove_labels=(config.labels.in_progress,),
            add_labels=(config.labels.pr_open,),
            branch=branch,
            base_branch="main",
        )
    ]

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"events": []}), encoding="utf-8")
    new_state = apply_fixes(
        gh, empty_state(), drift, config, repo_root=repo_root, state_path=state_path
    )

    # No vestigial PR, no push of the branch.
    assert not gh.prs_created
    remote_refs = _git(remote, "show-ref")
    assert "agent/issue-1241" not in remote_refs.stdout

    # The work already landed: do NOT relabel to ready (would loop) and do NOT
    # perform the in_progress -> pr_open swap.
    assert (1241, config.labels.ready) not in gh.labels_added
    assert (1241, config.labels.pr_open) not in gh.labels_added
    assert (1241, config.labels.in_progress) not in gh.labels_removed

    # The reconcile event for this item records the skip as a fix_action so the
    # skip is visible in the in-memory state ring too.
    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert any(
        e["payload"]["kind"] == "session_unpublished_work_salvaged"
        and any("salvage_skipped" in fa for fa in e["payload"]["fix_actions"])
        for e in reconcile_events
    )

    # Observable skip event recorded (``log_event`` writes to the SQLite event
    # log alongside state.json, not to the in-memory ``new_state`` ring -- same
    # channel as reconcile's other auxiliary events like ``pr_closing_ref_unlinked``).
    skip_events = [
        e for e in read_event_log(state_path) if e["kind"] == "salvage_skipped_already_landed"
    ]
    assert len(skip_events) == 1
    assert skip_events[0]["payload"]["issue_number"] == 1241
    assert skip_events[0]["payload"]["reason"] == REASON_ISSUE_CLOSED


def test_reconcile_salvage_skips_when_commits_reachable(tmp_path: Path) -> None:
    """Issue #1241 reachability path on the reconcile lane: the branch tip is
    an ancestor of origin/main (work landed via a sibling merge, main then
    advanced) while the issue is still OPEN and no merged PR binds to it. The
    reconcile lane skips with a ``salvage_skipped_superseded`` event and opens
    no PR."""
    from _reconcile_fixtures import FakeGitHub as ReconcileFakeGitHub

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 1241)
    _advance_main_past_branch(tmp_path, remote, repo_root, branch)

    config = OrchestratorConfig()
    gh = ReconcileFakeGitHub(
        prs=[],
        issues=[_issue(1241, [config.labels.in_progress], state="OPEN")],
        repo_root=repo_root,
        pr_create_return=999,
    )

    drift = [
        DriftItem(
            kind="session_unpublished_work_salvaged",
            issue_number=1241,
            pr_number=None,
            detail="salvage",
            fix_actions=("push", "pr_create"),
            remove_labels=(config.labels.in_progress,),
            add_labels=(config.labels.pr_open,),
            branch=branch,
            base_branch="main",
        )
    ]

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"events": []}), encoding="utf-8")
    new_state = apply_fixes(
        gh, empty_state(), drift, config, repo_root=repo_root, state_path=state_path
    )

    assert not gh.prs_created
    assert (1241, config.labels.ready) not in gh.labels_added
    assert (1241, config.labels.pr_open) not in gh.labels_added

    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert any(
        e["payload"]["kind"] == "session_unpublished_work_salvaged"
        and any("salvage_skipped" in fa for fa in e["payload"]["fix_actions"])
        for e in reconcile_events
    )

    superseded_events = [
        e for e in read_event_log(state_path) if e["kind"] == "salvage_skipped_superseded"
    ]
    assert len(superseded_events) == 1
    assert superseded_events[0]["payload"]["issue_number"] == 1241
    assert superseded_events[0]["payload"]["reason"] == REASON_COMMITS_REACHABLE


# ---------------------------------------------------------------------------
# event-kind registry contract: salvage_skip_event_kind only returns
# literals registered in _LEVEL_BY_KIND. This is the verification the
# _ALLOWED_UNRESOLVED_KIND_SITES entries in test_instrumentation.py point at:
# the dynamic ``salvage_skip_event_kind(skip_reason)`` call sites in
# dead_worker_reap._attempt_salvage and reconcile.apply_fixes resolve to one
# of these two registered literals, so they cannot escape the level registry.
# ---------------------------------------------------------------------------


def test_salvage_skip_event_kind_only_returns_registered_kinds() -> None:
    """Every possible return value of ``salvage_skip_event_kind`` is a literal
    registered in ``instrumentation._LEVEL_BY_KIND``. The two dynamic-kind
    emit sites (dead_worker_reap._attempt_salvage and reconcile.apply_fixes)
    call this mapper, so this test is what makes those sites safe despite
    being non-literal -- they cannot produce an unregistered kind."""
    from charlie_work.salvage_superseded import (
        REASON_COMMITS_REACHABLE,
        REASON_EMPTY_DIFF,
        REASON_ISSUE_CLOSED,
        REASON_PR_MERGED,
    )

    # Every reason check_salvage_superseded can return, plus None and an
    # unknown reason (the mapper's fall-through) -- exhaustive over the
    # mapper's input domain.
    for reason in (
        REASON_ISSUE_CLOSED,
        REASON_PR_MERGED,
        REASON_EMPTY_DIFF,
        REASON_COMMITS_REACHABLE,
        None,
        "unknown_future_reason",
    ):
        kind = salvage_skip_event_kind(reason)
        assert kind in _LEVEL_BY_KIND, (
            f"salvage_skip_event_kind({reason!r}) returned {kind!r}, which is "
            "not registered in _LEVEL_BY_KIND -- add it or narrow the mapper."
        )
