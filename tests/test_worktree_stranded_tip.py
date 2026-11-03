"""Issue #2904: rescue of a branch whose worktree directory no longer exists.

``create_worktree`` refuses to reset a branch whose commits are not on the
remote, and tries to rescue the work onto a ref first. The rescue staged the
working tree with ``cwd`` set to the worktree path, so with the directory gone
every call failed (``[WinError 267] The directory name is invalid``) and the
issue re-escalated on every requeue. A missing directory has no dirty content;
the committed tip is pushed to the salvage ref the refusal message names.

``tests/test_worktree.py`` is at its file-size ratchet mark, so these tests
live in their own module. They drive real git repos (a bare-ish remote and a
clone) through the shared ``_worktree_fixtures`` helpers -- no mocks on the
code under test.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from _worktree_fixtures import (
    REJECTED_PUSH_MESSAGE,
    _clone_repo,
    _git,
    _init_repo,
    _reject_pushes,
)
from _worktree_fixtures import _wt_scratch as _register_wt_scratch  # noqa: F401 -- registers the wt_scratch fixture: worker sandboxes redirect TEMP/TMPDIR deep inside the checkout, where tmp_path-derived repo roots overrun git's internal worktree-path buffer (``git worktree add`` exits 128)
from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.worktree import (
    RESCUE_REF_PREFIX,
    RescueCapture,
    WORKTREE_UNSAFE_KIND_LOCAL_COMMITS,
    WorktreeUnsafeError,
    _run_remote_captured,
    create_worktree,
)
from charlie_work.worktree_stranded_tip import (
    preserve_stranded_branch_tip,
    salvage_ref_for_branch,
)

ISSUE = 2904
# Substrings of the failure this issue removes. Neither may appear anywhere in
# a refusal: the rescue must not even attempt the working-tree capture.
WORKING_TREE_CAPTURE_FAILURES = ("WinError 267", "add/write-tree")


def _commit_in(worktree_path: Path, name: str = "work.txt") -> str:
    """Commit a file inside the worktree; return the new tip sha."""
    (worktree_path / name).write_text("stranded work\n", encoding="utf-8")
    _git(worktree_path, "add", name)
    _git(worktree_path, "commit", "-m", "worker commit on agent branch")
    return _git(worktree_path, "rev-parse", "HEAD").stdout.strip()


def _remote_ref_sha(remote: Path, ref: str) -> str | None:
    out = _git(remote, "for-each-ref", "--format=%(objectname)", ref).stdout.strip()
    return out or None


def _stranded_branch(wt_scratch: Path, *, delete_dir: bool = True) -> tuple[Path, Path, str, str]:
    """A repo with a remote and an agent branch ahead of it, worktree dir gone.

    Returns ``(remote, repo, branch, tip)``. By default the directory is deleted
    from under git (the registration is left stale, as after a crashed
    cleanup); ``delete_dir=False`` leaves the clean worktree in place.
    """
    remote = wt_scratch / "remote"
    _init_repo(remote)
    repo = wt_scratch / "repo"
    _clone_repo(remote, repo)
    branch = f"agent/issue-{ISSUE}-stranded"
    info = create_worktree(repo, branch, base_ref="HEAD", issue_number=ISSUE)
    tip = _commit_in(info.path)
    if delete_dir:
        shutil.rmtree(info.path)
    assert info.path.exists() is not delete_dir
    return remote, repo, branch, tip


def _events(repo: Path, config: OrchestratorConfig) -> list[dict]:
    state_file = runtime_paths(repo, config.runtime.state_dir).state_file
    return query_events(state_file, kind="worktree_rescue_captured")


def _direct(repo: Path, branch: str):
    return preserve_stranded_branch_tip(repo, branch, _run_remote_captured)


# --- acceptance: the dispatch-level behaviour -------------------------------


def test_missing_worktree_dir_pushes_tip_to_salvage_ref_and_dispatch_proceeds(
    wt_scratch: Path,
) -> None:
    """Acceptance 1: branch ahead of the remote, no worktree directory -> the
    salvage ref holds the tip, dispatch proceeds, no add/write-tree failure."""
    remote, repo, branch, tip = _stranded_branch(wt_scratch)
    main_tip = _git(repo, "rev-parse", "main").stdout.strip()
    config = OrchestratorConfig()
    salvage_ref = salvage_ref_for_branch(branch)
    assert _remote_ref_sha(remote, salvage_ref) is None

    info = create_worktree(repo, branch, base_ref="HEAD", issue_number=ISSUE, config=config)

    assert _remote_ref_sha(remote, salvage_ref) == tip
    assert info.rescue_capture == RescueCapture(ref_name=salvage_ref, commit_sha=tip)
    # The dispatch really went ahead: a fresh branch at the base, no stranded file.
    assert info.path.exists()
    assert _git(repo, "rev-parse", branch).stdout.strip() == main_tip
    assert not (info.path / "work.txt").exists()
    # The existing rescue event is emitted, naming the durable ref.
    events = _events(repo, config)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["rescue_ref"] == salvage_ref
    assert payload["commit_sha"] == tip
    assert payload["issue_number"] == ISSUE
    assert "not on remote branch" in payload["reason"]
    assert not any(marker in str(payload) for marker in WORKING_TREE_CAPTURE_FAILURES)


def test_redispatch_over_a_present_clean_worktree_reaches_the_same_push(
    wt_scratch: Path,
) -> None:
    """The directory need not be gone when dispatch starts. A fresh dispatch over
    a PRESENT worktree that is clean but holds unpushed commits removes the
    directory itself, then re-checks the branch -- so the missing-directory
    rescue runs there too. Before the fix that second check failed with
    ``[WinError 267]`` for exactly the same reason."""
    remote, repo, branch, tip = _stranded_branch(wt_scratch, delete_dir=False)
    salvage_ref = salvage_ref_for_branch(branch)
    config = OrchestratorConfig()

    info = create_worktree(repo, branch, base_ref="HEAD", issue_number=ISSUE, config=config)

    assert _remote_ref_sha(remote, salvage_ref) == tip
    assert info.rescue_capture is not None
    assert info.rescue_capture.ref_name == salvage_ref
    assert info.path.exists()
    assert not (info.path / "work.txt").exists()
    assert salvage_ref in [e["payload"]["rescue_ref"] for e in _events(repo, config)]


def test_push_failure_keeps_worktree_unsafe_error_with_push_error(wt_scratch: Path) -> None:
    """Acceptance 2: a failed push must not weaken the refusal, and the push's
    own error is in the message."""
    remote, repo, branch, tip = _stranded_branch(wt_scratch)
    _reject_pushes(remote)
    config = OrchestratorConfig()
    salvage_ref = salvage_ref_for_branch(branch)

    with pytest.raises(WorktreeUnsafeError) as excinfo:
        create_worktree(repo, branch, base_ref="HEAD", issue_number=ISSUE, config=config)

    message = str(excinfo.value)
    assert excinfo.value.kind == WORKTREE_UNSAFE_KIND_LOCAL_COMMITS
    assert "rescue capture failed" in message
    assert f"push to {salvage_ref} failed" in message
    assert REJECTED_PUSH_MESSAGE in message
    assert not any(marker in message for marker in WORKING_TREE_CAPTURE_FAILURES)
    # The refusal's own suggested push is the one that was attempted: the
    # message and the rescue share one spelling of the salvage ref.
    assert f"git push origin {tip}:{salvage_ref}" in message
    # Nothing was reset or published, and no rescue event claims durability.
    assert _git(repo, "rev-parse", branch).stdout.strip() == tip
    assert _remote_ref_sha(remote, salvage_ref) is None
    assert _events(repo, config) == []


def test_tip_already_on_a_pr_head_skips_the_push_and_reports_it(wt_scratch: Path) -> None:
    """A closed PR's commits live on ``refs/pull/<n>/head`` after its branch is
    deleted: the tip is already durable, so no salvage push is made -- but the
    rescue event still names where it was found."""
    remote, repo, branch, tip = _stranded_branch(wt_scratch)
    _git(repo, "push", "origin", f"{tip}:refs/pull/7/head")
    config = OrchestratorConfig()

    info = create_worktree(repo, branch, base_ref="HEAD", issue_number=ISSUE, config=config)

    assert info.rescue_capture == RescueCapture(ref_name="refs/pull/7/head", commit_sha=tip)
    assert _remote_ref_sha(remote, salvage_ref_for_branch(branch)) is None
    events = _events(repo, config)
    assert [e["payload"]["rescue_ref"] for e in events] == ["refs/pull/7/head"]
    assert info.path.exists()


def test_present_worktree_still_uses_the_working_tree_capture(
    wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance 3 guard: with the directory present the rescue is the
    unchanged ``git add``/``write-tree`` capture, never the stranded-tip push."""
    remote = wt_scratch / "remote"
    _init_repo(remote, bare=True)
    repo = wt_scratch / "repo"
    _clone_repo(remote, repo)
    branch = f"agent/issue-{ISSUE}-present"
    info1 = create_worktree(repo, branch, base_ref="origin/main")
    (info1.path / "wip.txt").write_text("dirty\n", encoding="utf-8")

    def _must_not_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("stranded-tip push ran for a present worktree directory")

    monkeypatch.setattr("charlie_work.worktree.preserve_stranded_branch_tip", _must_not_run)

    info2 = create_worktree(repo, branch, base_ref="origin/main", issue_number=ISSUE)

    assert info2.rescue_capture is not None
    assert info2.rescue_capture.ref_name is not None
    assert info2.rescue_capture.ref_name.startswith(RESCUE_REF_PREFIX)


# --- preserve_stranded_branch_tip: the helper's own contract -----------------


def test_tip_contained_in_an_advertised_remote_branch_skips_the_push(wt_scratch: Path) -> None:
    remote = wt_scratch / "remote"
    _init_repo(remote)
    repo = wt_scratch / "repo"
    _clone_repo(remote, repo)
    branch = f"agent/issue-{ISSUE}-contained"
    _git(repo, "checkout", "-b", branch)
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-m", "tip")
    tip = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "-b", "other")
    (repo / "b.txt").write_text("b\n", encoding="utf-8")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-m", "on top of the tip")
    _git(repo, "push", "origin", "other")

    outcome = _direct(repo, branch)

    assert outcome.error is None
    assert (outcome.ref_name, outcome.commit_sha, outcome.pushed) == (
        "refs/heads/other",
        tip,
        False,
    )
    assert _remote_ref_sha(remote, salvage_ref_for_branch(branch)) is None


def test_stale_tracking_ref_does_not_vouch_for_a_branch_the_remote_dropped(
    wt_scratch: Path,
) -> None:
    """``refs/remotes/origin/other`` still contains the tip locally, but the
    remote deleted ``other``: it is not proof of anything, so the tip is pushed."""
    remote = wt_scratch / "remote"
    _init_repo(remote)
    repo = wt_scratch / "repo"
    _clone_repo(remote, repo)
    branch = f"agent/issue-{ISSUE}-stale"
    _git(repo, "checkout", "-b", branch)
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-m", "tip")
    tip = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "push", "origin", f"{tip}:refs/heads/other")
    _git(repo, "fetch", "origin")
    _git(remote, "branch", "-D", "other")
    assert "refs/remotes/origin/other" in _git(repo, "for-each-ref", "refs/remotes").stdout

    outcome = _direct(repo, branch)

    salvage_ref = salvage_ref_for_branch(branch)
    assert outcome.error is None
    assert (outcome.ref_name, outcome.pushed) == (salvage_ref, True)
    assert _remote_ref_sha(remote, salvage_ref) == tip


def test_second_preserve_of_the_same_tip_reports_the_existing_salvage_ref(
    wt_scratch: Path,
) -> None:
    remote, repo, branch, tip = _stranded_branch(wt_scratch)
    salvage_ref = salvage_ref_for_branch(branch)

    first = _direct(repo, branch)
    second = _direct(repo, branch)

    assert (first.ref_name, first.pushed) == (salvage_ref, True)
    assert (second.ref_name, second.pushed) == (salvage_ref, False)
    assert second.error is None
    assert _remote_ref_sha(remote, salvage_ref) == tip


def test_existing_salvage_ref_at_another_tip_is_never_overwritten(wt_scratch: Path) -> None:
    """A plain push, never forced: an earlier salvage of different work is
    kept, and the rescue fails closed instead of replacing it."""
    remote, repo, branch, _tip = _stranded_branch(wt_scratch)
    salvage_ref = salvage_ref_for_branch(branch)
    # An unrelated earlier salvage: a root commit sharing no history with the tip.
    _git(repo, "checkout", "--orphan", "unrelated")
    (repo / "old.txt").write_text("earlier salvage\n", encoding="utf-8")
    _git(repo, "add", "old.txt")
    _git(repo, "commit", "-m", "earlier salvage")
    earlier = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "push", "origin", f"{earlier}:{salvage_ref}")

    outcome = _direct(repo, branch)

    assert outcome.ref_name is None
    assert outcome.error is not None
    assert f"push to {salvage_ref} failed" in outcome.error
    assert _remote_ref_sha(remote, salvage_ref) == earlier


def test_repo_without_origin_returns_an_error_value(wt_scratch: Path) -> None:
    """No remote means nothing can be made durable: the helper reports that as
    a value (it never raises), so the caller keeps its refusal."""
    repo = wt_scratch / "repo"
    _init_repo(repo)
    branch = f"agent/issue-{ISSUE}-no-origin"
    info = create_worktree(repo, branch, base_ref="HEAD", issue_number=ISSUE)
    tip = _commit_in(info.path)

    outcome = _direct(repo, branch)

    assert outcome.ref_name is None
    assert outcome.commit_sha == tip
    assert outcome.error is not None
    assert "push to refs/heads/salvage/" in outcome.error


def test_unresolvable_branch_returns_an_error_value(wt_scratch: Path) -> None:
    repo = wt_scratch / "repo"
    _init_repo(repo)

    outcome = _direct(repo, "agent/issue-2904-never-existed")

    assert (outcome.ref_name, outcome.commit_sha) == (None, None)
    assert outcome.error is not None
    assert "refs/heads/agent/issue-2904-never-existed" in outcome.error


def test_invalid_branch_name_never_reaches_git(wt_scratch: Path) -> None:
    repo = wt_scratch / "repo"
    _init_repo(repo)

    outcome = _direct(repo, "bad branch;rm -rf")

    assert (outcome.ref_name, outcome.commit_sha) == (None, None)
    assert outcome.error is not None
