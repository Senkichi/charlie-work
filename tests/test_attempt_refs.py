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
from unittest import mock

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


def _seed_branch_commit(repo_root: Path, branch: str, name: str, content: str) -> str:
    """Create ``branch`` one commit ahead of ``main`` in one ``git fast-import``
    spawn and return the new tip SHA.

    Replaces the ``checkout -b`` + ``add`` + ``commit`` + ``rev-parse`` quartet
    (four spawns, plus an index rewrite and a working-tree write) for a test that
    only needs the branch *ref* to exist: ``snapshot_attempt_ref`` resolves the
    branch by name and never reads HEAD, the index or the working tree. HEAD
    therefore stays on ``main``. ``--export-marks`` hands back the SHA without a
    ``rev-parse`` (same shape as ``_backfill_gate_fixtures._make_repo_with_fix``,
    issue #2605; ledger precedent #2569). Issue #2936.
    """
    marks_path = repo_root.parent / f"{repo_root.name}.marks"
    subprocess.run(
        ["git", "fast-import", "--quiet", f"--export-marks={marks_path}"],
        cwd=repo_root,
        # input as bytes: text=True would translate \n to \r\n on Windows
        # stdin, and git parses the \r as part of each command line.
        input=(
            f"commit refs/heads/{branch}\n"
            "mark :1\n"
            "committer Test User <test@example.test> 0 +0000\n"
            "data <<MSG\n"
            f"add {name}\n"
            "MSG\n"
            "from refs/heads/main\n"
            f"M 100644 inline {name}\n"
            "data <<EOT\n"
            f"{content}"
            "EOT\n"
        ).encode(),
        check=True,
        capture_output=True,
    )
    return marks_path.read_text(encoding="utf-8").split()[1]  # ":1 <sha>"


def test_seed_branch_commit_is_one_spawn_one_commit_ahead_of_main(tmp_path: Path) -> None:
    """Issue #2936 regression pin: the preserves-tip fixture must build its branch
    in a single child process, as a real commit whose only parent is ``main``.

    The ledger flagged ``test_snapshot_attempt_ref_preserves_tip`` at 2.3x
    baseline (0.583 s -> 1.313 s median on CI) with no change to this code path:
    the first slow SHA touches only ``aggregate_job_heal``/``infra_rerun_refusal``.
    The repo's documented mechanism for that shape is a loaded runner inflating
    per-spawn and per-file cost (#2577, #2605), so the fix is fewer call-phase
    spawns -- nine became six. Counting ``Popen`` (which ``subprocess.run``
    goes through) catches a regression back to ``_git``/``run_captured`` calls
    as well as a second ``fast-import``.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    main_tip = _git(repo_root, "rev-parse", "main").stdout.strip()

    with mock.patch.object(subprocess, "Popen", wraps=subprocess.Popen) as popen:
        tip = _seed_branch_commit(repo_root, "agent/issue-1", "work.txt", "some work\n")

    assert popen.call_count == 1
    assert _git(repo_root, "rev-parse", "agent/issue-1").stdout.strip() == tip
    # `--parents -n1` prints "<commit> <parent>..."; a single parent that is main's tip.
    assert _git(repo_root, "rev-list", "--parents", "-n1", tip).stdout.split() == [tip, main_tip]
    assert _git(repo_root, "show", f"{tip}:work.txt").stdout == "some work\n"
    # The seed must not move HEAD or dirty the tree the other tests in this module rely on.
    assert _git(repo_root, "symbolic-ref", "--short", "HEAD").stdout.strip() == "main"
    assert _git(repo_root, "status", "--porcelain").stdout == ""


def test_snapshot_attempt_ref_preserves_tip(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    tip = _seed_branch_commit(repo_root, "agent/issue-1", "work.txt", "some work\n")

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
