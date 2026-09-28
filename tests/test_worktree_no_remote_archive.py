"""Issue #1944: no-remote repos archive diverged agent branches on requeue.

On a repo with no origin remote, an agent branch's commits past the default
branch can never become "pushed" — there is nowhere to push to. Before this
fix, ``create_worktree``'s refuse-to-reset guard classified exactly that
shape as ``worktree_unsafe_local_commits``, escalating the issue to
``agent:human-needed`` on every requeue until someone deleted the branch by
hand (mdls #144). The fix archives the unreachable tip to a local
``archive/<branch>-<utc-date>`` branch and lets the reset proceed; repos
WITH a remote keep refusing, because an unpushed commit there can still be
salvaged by a real push.

``tests/test_worktree.py`` is over its attachment-contract member ceiling,
so these tests live in their own module. They exercise real git repos — no
mocks — via the shared ``_worktree_fixtures`` helpers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from _worktree_fixtures import _clone_repo, _git, _init_repo
from _worktree_fixtures import _wt_scratch as _register_wt_scratch  # noqa: F401 -- registers the wt_scratch fixture: worker sandboxes redirect TEMP/TMPDIR deep inside the checkout, where tmp_path-derived repo roots overrun git's internal worktree-path buffer (``git worktree add`` exits 128)
from charlie_work.config import DispatchConfig, OrchestratorConfig
from charlie_work.instrumentation import query_events
from charlie_work.worktree import WorktreeUnsafeError, create_worktree, remove_worktree


def _commit_in(worktree_path: Path, name: str = "work.txt") -> str:
    """Commit a file inside the worktree; return the new tip sha."""
    (worktree_path / name).write_text("local work\n", encoding="utf-8")
    _git(worktree_path, "add", name)
    _git(worktree_path, "commit", "-m", "worker commit on agent branch")
    return _git(worktree_path, "rev-parse", "HEAD").stdout.strip()


def _archive_refs(repo_root: Path, branch: str) -> dict[str, str]:
    """Map each ``archive/<branch>-*`` branch name to the sha it points at."""
    out = _git(
        repo_root,
        "for-each-ref",
        "--format=%(refname) %(objectname)",
        "refs/heads/archive/",
    ).stdout
    prefix = f"archive/{branch}-"
    refs: dict[str, str] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        refname, sha = line.split(" ", 1)
        name = refname[len("refs/heads/") :]
        if name.startswith(prefix):
            refs[name] = sha
    return refs


def _today() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")


def test_no_remote_diverged_branch_is_archived_and_recreated(wt_scratch: Path) -> None:
    """Spec AC 1: a no-remote repo with a diverged agent branch gets an
    archive ref at the old tip plus a fresh branch — no escalation.

    This is the mdls #144 shape exactly: the rebase was abandoned, the
    worktree is gone, only the diverged branch remains, and the requeued
    issue hits fresh dispatch.
    """
    repo_root = wt_scratch / "repo"
    _init_repo(repo_root)  # no origin remote
    branch = "agent/issue-144-example"
    main_tip = _git(repo_root, "rev-parse", "main").stdout.strip()

    info1 = create_worktree(repo_root, branch, base_ref="HEAD", issue_number=144)
    old_tip = _commit_in(info1.path)
    assert old_tip != main_tip
    # Abandoned worktree: removed, but the diverged branch survives.
    assert remove_worktree(repo_root, info1.path)
    assert not info1.path.exists()

    # Fresh dispatch on the requeued issue must NOT raise
    # WorktreeUnsafeError — it archives the old tip and recreates the
    # branch from the default branch.
    info2 = create_worktree(repo_root, branch, base_ref="HEAD", issue_number=144)

    archives = _archive_refs(repo_root, branch)
    assert archives == {f"archive/{branch}-{_today()}": old_tip}
    assert _git(repo_root, "rev-parse", branch).stdout.strip() == main_tip
    assert info2.path.exists()
    assert not (info2.path / "work.txt").exists()


def test_no_remote_diverged_worktree_present_is_archived_and_pruned(
    wt_scratch: Path,
) -> None:
    """Same as above but the worker's worktree directory is still present
    (clean tree, diverged commits): reclaim archives the tip, removes the
    stale worktree, and recreates at the base."""
    repo_root = wt_scratch / "repo"
    _init_repo(repo_root)
    branch = "agent/issue-200-live-worktree"
    main_tip = _git(repo_root, "rev-parse", "main").stdout.strip()

    info1 = create_worktree(repo_root, branch, base_ref="HEAD", issue_number=200)
    old_tip = _commit_in(info1.path)

    info2 = create_worktree(repo_root, branch, base_ref="HEAD", issue_number=200)

    assert info2.reclaimed == "pruned"
    archives = _archive_refs(repo_root, branch)
    assert archives == {f"archive/{branch}-{_today()}": old_tip}
    assert _git(repo_root, "rev-parse", branch).stdout.strip() == main_tip
    assert info2.path.exists()


def test_no_remote_existing_identical_archive_ref_is_reused(wt_scratch: Path) -> None:
    """Spec: skip creating the archive ref when an identical one already
    exists — a second probe over the same tip must not mint a ``-2`` name."""
    repo_root = wt_scratch / "repo"
    _init_repo(repo_root)
    branch = "agent/issue-201-idempotent"
    main_tip = _git(repo_root, "rev-parse", "main").stdout.strip()

    info1 = create_worktree(repo_root, branch, base_ref="HEAD", issue_number=201)
    old_tip = _commit_in(info1.path)
    assert remove_worktree(repo_root, info1.path)

    # The archive ref already exists at the same tip (e.g. an earlier probe
    # pass created it before a later failure aborted the dispatch).
    _git(repo_root, "branch", f"archive/{branch}-{_today()}", old_tip)

    info2 = create_worktree(repo_root, branch, base_ref="HEAD", issue_number=201)

    # Still exactly one archive ref — the identical one was reused, not
    # duplicated under a suffix.
    assert _archive_refs(repo_root, branch) == {f"archive/{branch}-{_today()}": old_tip}
    assert _git(repo_root, "rev-parse", branch).stdout.strip() == main_tip
    assert info2.path.exists()


def test_no_remote_archive_name_collision_gets_suffix(wt_scratch: Path) -> None:
    """A same-named archive ref at a DIFFERENT tip must not be overwritten —
    the new archive takes a ``-2`` suffix instead."""
    repo_root = wt_scratch / "repo"
    _init_repo(repo_root)
    branch = "agent/issue-202-collision"
    main_tip = _git(repo_root, "rev-parse", "main").stdout.strip()

    info1 = create_worktree(repo_root, branch, base_ref="HEAD", issue_number=202)
    old_tip = _commit_in(info1.path)
    assert remove_worktree(repo_root, info1.path)

    # Pre-existing archive ref for a different (earlier) tip.
    _git(repo_root, "branch", f"archive/{branch}-{_today()}", main_tip)

    create_worktree(repo_root, branch, base_ref="HEAD", issue_number=202)

    archives = _archive_refs(repo_root, branch)
    assert archives == {
        f"archive/{branch}-{_today()}": main_tip,
        f"archive/{branch}-{_today()}-2": old_tip,
    }


def test_no_remote_archive_failure_falls_through_to_refusal(wt_scratch: Path) -> None:
    """Fail-closed: when the archive ref itself cannot be created, the reset
    must still refuse — a failed archive never permits discarding the tip.

    A branch literally named ``archive`` pins ``refs/heads/archive`` as a
    loose ref, so every ``archive/<branch>-<date>`` create fails with a ref
    D/F conflict — a deterministic archive failure with no mocks. The
    fall-through to capture-or-refuse is the core safety property of the
    #1944 fix.
    """
    repo_root = wt_scratch / "repo"
    _init_repo(repo_root)  # no origin remote
    branch = "agent/issue-206-archive-failure"

    info1 = create_worktree(repo_root, branch, base_ref="HEAD", issue_number=206)
    old_tip = _commit_in(info1.path)
    assert remove_worktree(repo_root, info1.path)
    assert not info1.path.exists()

    # refs/heads/archive now exists as a ref, so refs/heads/archive/<x> is
    # un-creatable (D/F conflict) — _archive_unreachable_branch_tip must
    # return None and the reset must refuse rather than reset anyway.
    _git(repo_root, "branch", "archive")

    with pytest.raises(WorktreeUnsafeError, match="local commit"):
        create_worktree(repo_root, branch, base_ref="HEAD", issue_number=206)

    # The refusal preserved everything: the diverged tip is intact on the
    # agent branch, no archive ref was minted, and no new worktree exists.
    assert _git(repo_root, "rev-parse", branch).stdout.strip() == old_tip
    assert _archive_refs(repo_root, branch) == {}
    assert not info1.path.exists()


def test_no_remote_archive_kill_switch_off_still_refuses(wt_scratch: Path) -> None:
    """The config kill switch restores refuse-and-escalate when set false."""
    repo_root = wt_scratch / "repo"
    _init_repo(repo_root)
    branch = "agent/issue-203-kill-switch"

    info1 = create_worktree(repo_root, branch, base_ref="HEAD", issue_number=203)
    _commit_in(info1.path)
    assert remove_worktree(repo_root, info1.path)

    config = OrchestratorConfig(dispatch=DispatchConfig(archive_unreachable_local_commits=False))
    with pytest.raises(WorktreeUnsafeError, match="local commit"):
        create_worktree(repo_root, branch, base_ref="HEAD", issue_number=203, config=config)

    # Nothing was archived and the diverged branch is untouched.
    assert _archive_refs(repo_root, branch) == {}
    assert (
        _git(repo_root, "rev-parse", branch).stdout.strip()
        != _git(repo_root, "rev-parse", "main").stdout.strip()
    )


def test_with_remote_unpushed_commits_still_refuses(wt_scratch: Path) -> None:
    """Spec AC 2: a repo WITH an origin remote keeps refusing — unpushed
    commits there can be salvaged by a real push, so no archive shortcut."""
    remote_repo = wt_scratch / "remote"
    _init_repo(remote_repo)
    repo_root = wt_scratch / "repo"
    _clone_repo(remote_repo, repo_root)
    branch = "agent/issue-204-remote"
    main_tip = _git(repo_root, "rev-parse", "main").stdout.strip()

    info1 = create_worktree(repo_root, branch, base_ref="HEAD", issue_number=204)
    old_tip = _commit_in(info1.path)
    assert remove_worktree(repo_root, info1.path)

    with pytest.raises(WorktreeUnsafeError, match="local commit"):
        create_worktree(repo_root, branch, base_ref="HEAD", issue_number=204)

    # Refusal preserved everything: no archive ref, diverged branch intact.
    assert _archive_refs(repo_root, branch) == {}
    assert _git(repo_root, "rev-parse", branch).stdout.strip() == old_tip
    assert old_tip != main_tip


def test_no_remote_archive_emits_event(wt_scratch: Path) -> None:
    """The archive decision is recorded as ``worktree_local_commits_archived``
    with the branch, archive ref, and tip sha."""
    repo_root = wt_scratch / "repo"
    _init_repo(repo_root)
    branch = "agent/issue-205-event"

    info1 = create_worktree(repo_root, branch, base_ref="HEAD", issue_number=205)
    old_tip = _commit_in(info1.path)
    assert remove_worktree(repo_root, info1.path)

    config = OrchestratorConfig()
    create_worktree(repo_root, branch, base_ref="HEAD", issue_number=205, config=config)

    state_file = repo_root / ".var" / "charlie-work" / "state.json"
    events = query_events(state_file, kind="worktree_local_commits_archived")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["branch"] == branch
    assert payload["archive_ref"] == f"archive/{branch}-{_today()}"
    assert payload["tip_sha"] == old_tip
    assert payload["issue_number"] == 205
