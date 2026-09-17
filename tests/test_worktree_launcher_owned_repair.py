"""Pre-merge repair of launcher-owned residue on the rework branch (#1688).

Split out of ``test_worktree.py`` under the issue #1442 file-size ratchet:
the #1688 fix's regression tests landed in the over-cap monolith and tripped
its recorded mark, so they live here — the same sibling-module move the
collect-only gate's ``missing_sibling`` clause exists to allow (leaf names
reappear under ``tests/``, so nothing reads as dropped). The helpers they
need (``_init_repo``, ``_git``) come from ``tests/_worktree_fixtures.py``.

Scope: ``_merge_update_rework_branch``'s pre-merge repair eligibility covers
the launcher-owned PR-body FILE family (tracked-modified and untracked
classes) but never the launcher-owned DIRECTORIES, which can be junctions or
other reparse points on Windows.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from _worktree_fixtures import _git, _init_repo
from charlie_work.worktree import (
    ReworkBranchConflictError,
    _merge_update_rework_branch,
)


def test_merge_update_rework_branch_restores_modified_launcher_owned_pr_body_and_retries(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Issue #1688 regression — the exact #1477 stuck state: a *tracked*
    ``PR_BODY.md`` with a local uncommitted modification collides with the
    base's own moving copy of the same file. Before the fix every rework
    pre-merge failed identically (no MERGE_HEAD) because the repair path
    recognized only declared scaffolding while the dirty check already
    ignored launcher residue — so the branch wedged and the janitor kept
    escalating ``no_op_rework``.

    ``PR_BODY.md`` is launcher-owned, never orchestrator-declared, so no
    ``injected_paths``/``materialize_dirs`` are passed. The local
    modification must be discarded via ``git checkout HEAD --`` and the
    merge retried once, succeeding — and the repair's log output must name
    the file it dropped so the discard is visible after the fact rather
    than silently lost.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Common ancestor: both branches start out tracking the launcher-owned
    # scratch file (the shape #1477 found on the reused rework worktree).
    (repo_root / "PR_BODY.md").write_text("ancestor pr body v0\n", encoding="utf-8")
    _git(repo_root, "add", "PR_BODY.md")
    _git(repo_root, "commit", "-m", "add PR_BODY.md to ancestor")

    _git(repo_root, "checkout", "-b", "feature")
    (repo_root / "work.txt").write_text("worker output\n", encoding="utf-8")
    _git(repo_root, "add", "work.txt")
    _git(repo_root, "commit", "-m", "feature work")

    _git(repo_root, "checkout", "main")
    # The base's own copy keeps moving — salvage commits rewrite it.
    (repo_root / "PR_BODY.md").write_text("main pr body v1\n", encoding="utf-8")
    _git(repo_root, "add", "PR_BODY.md")
    _git(repo_root, "commit", "-m", "base rewrites PR_BODY.md")

    _git(repo_root, "checkout", "feature")
    # The launcher/worker rewrote the draft in place, uncommitted — a
    # locally modified TRACKED file, the dirt that wedged #1477.
    (repo_root / "PR_BODY.md").write_text("locally drafted pr body\n", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="charlie_work.worktree"):
        result = _merge_update_rework_branch(repo_root, repo_root, "feature", "main")

    assert result is None
    merge_head_check = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", "MERGE_HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert merge_head_check.returncode != 0
    # The local draft was discarded and the merge landed the base's copy.
    assert (repo_root / "PR_BODY.md").read_text(encoding="utf-8") == "main pr body v1\n"
    # An unrelated worker-authored tracked file must survive untouched.
    assert (repo_root / "work.txt").read_text(encoding="utf-8") == "worker output\n"
    # The repair names the file it discarded — not a silent drop.
    assert any("PR_BODY.md" in record.getMessage() for record in caplog.records)


def test_merge_update_rework_branch_clears_untracked_launcher_owned_pr_body_and_retries(
    tmp_path: Path,
) -> None:
    """Untracked-class companion to the #1688 regression: an *untracked*
    ``PR_BODY.md`` shadowing a path the base now tracks is likewise
    repair-eligible (``git clean`` removal), not a pre-merge escalation.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    _git(repo_root, "checkout", "-b", "feature")
    (repo_root / "work.txt").write_text("worker output\n", encoding="utf-8")
    _git(repo_root, "add", "work.txt")
    _git(repo_root, "commit", "-m", "feature work")

    _git(repo_root, "checkout", "main")
    (repo_root / "PR_BODY.md").write_text("main pr body v1\n", encoding="utf-8")
    _git(repo_root, "add", "PR_BODY.md")
    _git(repo_root, "commit", "-m", "base adds tracked PR_BODY.md")

    _git(repo_root, "checkout", "feature")
    # An untracked local draft shadows the base-tracked path.
    (repo_root / "PR_BODY.md").write_text("locally drafted pr body\n", encoding="utf-8")

    result = _merge_update_rework_branch(repo_root, repo_root, "feature", "main")

    assert result is None
    # The merge landed the base's tracked copy over the removed draft.
    assert (repo_root / "PR_BODY.md").read_text(encoding="utf-8") == "main pr body v1\n"
    assert (repo_root / "work.txt").read_text(encoding="utf-8") == "worker output\n"


def test_merge_update_rework_branch_launcher_owned_dir_blocker_raises_pre_merge(
    tmp_path: Path,
) -> None:
    """Issue #1688 directory control: repair eligibility extends to the
    launcher-owned PR-body FILE family only — never to paths under the
    launcher-owned DIRECTORIES (``.devin/``, ``.git_worktree_dir/``). A
    launcher-owned directory can be a junction or other reparse point on
    Windows, and a ``git clean`` sweep that follows one escapes the
    worktree.

    An untracked collision under ``.devin/`` — launcher-owned but not the
    PR-body file family and not declared scaffolding — must still escalate
    as ``stage="pre_merge"`` with nothing under the directory removed. This
    fails if repair eligibility is widened to the whole
    ``_launcher_owned_matcher``.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    _git(repo_root, "checkout", "-b", "feature")
    (repo_root / "work.txt").write_text("worker output\n", encoding="utf-8")
    _git(repo_root, "add", "work.txt")
    _git(repo_root, "commit", "-m", "feature work")

    _git(repo_root, "checkout", "main")
    (repo_root / ".devin" / "prompts").mkdir(parents=True)
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "main prompts v1\n", encoding="utf-8"
    )
    _git(repo_root, "add", ".devin/prompts/worker.md")
    _git(repo_root, "commit", "-m", "base adds tracked prompt under .devin")

    _git(repo_root, "checkout", "feature")
    # An UNTRACKED local copy under a launcher-owned directory shadows the
    # base-tracked path — the removal hazard the narrow scoping exists for.
    (repo_root / ".devin" / "prompts").mkdir(parents=True)
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "local shim copy\n", encoding="utf-8"
    )

    with pytest.raises(ReworkBranchConflictError) as exc_info:
        _merge_update_rework_branch(repo_root, repo_root, "feature", "main")

    assert exc_info.value.stage == "pre_merge"
    assert ".devin/prompts/worker.md" in exc_info.value.conflicted_paths
    # Nothing under the launcher-owned directory was removed.
    shim_copy = repo_root / ".devin" / "prompts" / "worker.md"
    assert shim_copy.read_text(encoding="utf-8") == "local shim copy\n"
