"""Unit tests for attempt_refs.py (issue #261).

Covers the plumbing in isolation (snapshot creation, attempt numbering,
listing, ahead-of-main computation, graceful failure) — the end-to-end
redispatch integration (attempt ref survives a real ``create_worktree``
branch reset) lives in test_worktree.py alongside the other recovery-path
tests it shares fixtures with.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from _worktree_fixtures import _init_repo as _init_repo_templated
from charlie_work.attempt_refs import (
    ATTEMPT_REF_PREFIX,
    AttemptSnapshot,
    list_attempt_refs,
    snapshot_attempt_ref,
)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(repo_root: Path) -> None:
    """A one-commit ``main`` repo with the test identity — byte-identical to the
    ``init``/``config``/``commit`` spawn sequence this helper used to run inline.

    Delegates to ``_worktree_fixtures._init_repo`` so the repo materializes
    from the per-process ``plain`` git template (``shutil.copytree``, no
    subprocesses — HS-CW-4, ``tests/_git_templates.py``): the spawn cost,
    amplified on the Windows CI host, is what the ledger flagged at ~3.4x
    baseline (issue #2462, same fix class as #2387/#2425).
    """
    _init_repo_templated(repo_root)


def test_init_repo_goes_through_the_git_template(tmp_path: Path) -> None:
    """Issue #2462 regression pin: ``_init_repo`` must materialize through the
    per-process ``plain`` git template (``_git_templates``) instead of
    re-running the ``init``/``config``/``commit`` spawn sequence per call —
    the ledger measured
    ``test_snapshot_attempt_ref_no_commits_returns_all_none`` at ~3.4x
    baseline because each of this module's tests paid the full spawn cost on
    the Windows CI host. Mirrors ``test_local_issues_loop_gates``'s #2425 pin.
    """
    import _git_templates
    from _worktree_fixtures import _git as wt_git

    before = _git_templates._REGISTRY.materialized["plain"]
    repo = tmp_path / "repo"
    _init_repo(repo)
    assert _git_templates._REGISTRY.materialized["plain"] == before + 1
    assert wt_git(repo, "rev-parse", "--verify", "main").stdout.strip()


def _commit(repo_root: Path, name: str, content: str) -> str:
    (repo_root / name).write_text(content, encoding="utf-8")
    _git(repo_root, "add", name)
    _git(repo_root, "commit", "-m", f"add {name}")
    return _git(repo_root, "rev-parse", "HEAD").stdout.strip()


def test_snapshot_attempt_ref_preserves_tip(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    _git(repo_root, "checkout", "-b", "agent/issue-1")
    tip = _commit(repo_root, "work.txt", "some work\n")

    snapshot = snapshot_attempt_ref(repo_root, "agent/issue-1", issue_number=1, base_ref="main")

    assert snapshot.error is None
    assert snapshot.old_tip == tip
    assert snapshot.ref_name == f"{ATTEMPT_REF_PREFIX}/issue-1/attempt-1"
    assert snapshot.ahead_of_main_count == 1

    resolved = _git(repo_root, "rev-parse", snapshot.ref_name).stdout.strip()
    assert resolved == tip


def test_snapshot_attempt_ref_no_commits_returns_all_none(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    # Branch that does not resolve to a commit (never created).
    snapshot = snapshot_attempt_ref(repo_root, "does-not-exist", issue_number=1, base_ref="main")

    assert snapshot.ref_name is None
    assert snapshot.old_tip is None
    assert snapshot.ahead_of_main_count is None
    assert snapshot.error is None  # not an error — simply nothing to preserve


def test_snapshot_attempt_number_increments_across_calls(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    _git(repo_root, "checkout", "-b", "agent/issue-7")
    tip1 = _commit(repo_root, "attempt1.txt", "attempt 1\n")

    first = snapshot_attempt_ref(repo_root, "agent/issue-7", issue_number=7, base_ref="main")
    assert first.ref_name == f"{ATTEMPT_REF_PREFIX}/issue-7/attempt-1"
    assert first.old_tip == tip1

    # Simulate a second attempt: reset the branch and commit again.
    tip2 = _commit(repo_root, "attempt2.txt", "attempt 2\n")
    second = snapshot_attempt_ref(repo_root, "agent/issue-7", issue_number=7, base_ref="main")
    assert second.ref_name == f"{ATTEMPT_REF_PREFIX}/issue-7/attempt-2"
    assert second.old_tip == tip2
    assert second.old_tip != first.old_tip

    refs = list_attempt_refs(repo_root, 7)
    assert refs == (
        f"{ATTEMPT_REF_PREFIX}/issue-7/attempt-1",
        f"{ATTEMPT_REF_PREFIX}/issue-7/attempt-2",
    )


def test_list_attempt_refs_empty_when_none_exist(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    assert list_attempt_refs(repo_root, 999) == ()


def test_snapshot_attempt_ref_never_raises_on_non_git_directory(tmp_path: Path) -> None:
    """A repo_root that is not a git repository must degrade to an
    AttemptSnapshot with old_tip=None/error set, never raise — attempt
    preservation is fire-and-forget insurance around a redispatch and must
    never itself become the reason a redispatch fails.
    """
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()

    snapshot = snapshot_attempt_ref(not_a_repo, "some-branch", issue_number=1, base_ref="main")

    assert isinstance(snapshot, AttemptSnapshot)
    assert snapshot.ref_name is None
    assert snapshot.old_tip is None


def test_list_attempt_refs_never_raises_on_non_git_directory(tmp_path: Path) -> None:
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()

    assert list_attempt_refs(not_a_repo, 1) == ()
