"""Declared-scaffolding collision clearing, restore and repair around the rework
merge-update."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from _worktree_fixtures import _git, _init_repo, _seed_staged_addition_bubble
from charlie_work.worktree import (
    ReworkBranchConflictError,
    _clear_declared_scaffolding_collisions,
    _eligible_for_scaffolding_repair,
    _merge_update_rework_branch,
    _modified_paths_overwritten_by_ref,
    _repair_declared_scaffolding_blockers,
    _restore_declared_scaffolding_modifications,
)


def test_merge_update_rework_branch_clears_declared_scaffolding_and_retries(
    tmp_path: Path,
) -> None:
    """Case (c) with a declared-scaffolding collision: an untracked copy of a
    now-base-tracked path, left by the orchestrator's own materializer, is
    cleared and the merge retried once — succeeding, because there was never
    a real conflict between the branch and the base.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    _git(repo_root, "checkout", "-b", "feature")
    (repo_root / "work.txt").write_text("worker output\n", encoding="utf-8")
    _git(repo_root, "add", "work.txt")
    _git(repo_root, "commit", "-m", "feature work")

    _git(repo_root, "checkout", "main")
    (repo_root / "prompts").mkdir()
    (repo_root / "prompts" / "PROMPT.md").write_text("base prompt\n", encoding="utf-8")
    _git(repo_root, "add", "prompts/PROMPT.md")
    _git(repo_root, "commit", "-m", "add tracked prompt to main")

    _git(repo_root, "checkout", "feature")
    # Simulate orchestrator scaffolding materialized into the worktree ahead
    # of this pre-merge: an UNTRACKED copy at a path the base now tracks.
    (repo_root / "prompts").mkdir()
    (repo_root / "prompts" / "PROMPT.md").write_text("scaffolding prompt\n", encoding="utf-8")
    # An untracked, UNDECLARED file that sits outside the blocking set. It is
    # not tracked by main (so `git clean` *can* delete it, unlike work.txt
    # below), which is what makes it able to catch a cleanup that drops the
    # `-- <blocking_paths>` pathspec and sweeps the whole worktree instead of
    # only the declared collision.
    (repo_root / "scratch-notes.txt").write_text("worker scratch\n", encoding="utf-8")

    result = _merge_update_rework_branch(
        repo_root,
        repo_root,
        "feature",
        "main",
        injected_paths=("prompts/PROMPT.md",),
    )

    assert result is None
    # The merge succeeded, so the base's tracked copy is now checked out.
    assert (repo_root / "prompts" / "PROMPT.md").read_text(encoding="utf-8") == "base prompt\n"
    # An unrelated worker-authored (tracked) file must survive untouched.
    assert (repo_root / "work.txt").read_text(encoding="utf-8") == "worker output\n"
    # An unrelated worker-authored (untracked, undeclared) file must also
    # survive: only the declared collision is in scope for the cleanup.
    assert (repo_root / "scratch-notes.txt").read_text(encoding="utf-8") == "worker scratch\n"


def test_merge_update_rework_branch_undeclared_blocker_raises_pre_merge(
    tmp_path: Path,
) -> None:
    """An untracked file the base tracks, but which was never declared as the
    orchestrator's own scaffolding (via ``injected_paths``/``materialize_dirs``),
    must not be deleted. Escalating is still correct — but as
    ``stage="pre_merge"`` (not a content conflict), and with the file left in
    place on disk.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    _git(repo_root, "checkout", "-b", "feature")
    (repo_root / "work.txt").write_text("worker output\n", encoding="utf-8")
    _git(repo_root, "add", "work.txt")
    _git(repo_root, "commit", "-m", "feature work")

    _git(repo_root, "checkout", "main")
    (repo_root / "shared.txt").write_text("base version\n", encoding="utf-8")
    _git(repo_root, "add", "shared.txt")
    _git(repo_root, "commit", "-m", "add tracked shared.txt to main")

    _git(repo_root, "checkout", "feature")
    # An untracked collision that is NOT declared scaffolding, e.g. a
    # worker-authored file that happens to share a base-tracked path.
    (repo_root / "shared.txt").write_text("worker's own copy\n", encoding="utf-8")

    with pytest.raises(ReworkBranchConflictError) as exc_info:
        _merge_update_rework_branch(repo_root, repo_root, "feature", "main")

    assert exc_info.value.stage == "pre_merge"
    assert "shared.txt" in exc_info.value.conflicted_paths
    # Nothing declared as scaffolding was involved, so nothing is deleted.
    assert (repo_root / "shared.txt").read_text(encoding="utf-8") == "worker's own copy\n"


def test_clear_declared_scaffolding_collisions_refuses_venv_paths(tmp_path: Path) -> None:
    """``.venv`` must never be swept by the pre-merge scaffolding cleanup, even
    if it were somehow declared as injected/materialize scaffolding — that
    path is a junction into the SHARED virtualenv on this host, and
    ``git clean -f -d`` would happily follow it into every other worktree.

    Includes a positive control on an ordinary (non-``.venv``) declared
    collision in the same repo: without it, "returns False and removes
    nothing" for the .venv case would be equally consistent with the
    function being unconditionally broken rather than specifically refusing
    ``.venv``.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    venv_dir = repo_root / ".venv"
    venv_dir.mkdir()
    marker = venv_dir / "pyvenv.cfg"
    marker.write_text("home = fake\n", encoding="utf-8")

    cleared = _clear_declared_scaffolding_collisions(
        repo_root,
        (".venv/pyvenv.cfg",),
        (".venv",),
        (),
    )

    assert cleared is False
    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "home = fake\n"

    # Positive control: an ordinary declared collision, untracked and
    # outside .venv, IS cleared. This proves False above means "refused",
    # not "the function can never return True."
    (repo_root / "prompts").mkdir()
    ordinary = repo_root / "prompts" / "PROMPT.md"
    ordinary.write_text("scaffolding prompt\n", encoding="utf-8")

    cleared_ordinary = _clear_declared_scaffolding_collisions(
        repo_root,
        ("prompts/PROMPT.md",),
        ("prompts/PROMPT.md",),
        (),
    )

    assert cleared_ordinary is True
    assert not ordinary.exists()


def test_merge_update_rework_branch_restores_modified_declared_scaffolding_and_retries(
    tmp_path: Path,
) -> None:
    """Case (c) with a *modified* (not untracked) scaffolding collision: the
    base ref deletes a tracked scaffolding path that the branch still tracks
    and that is locally modified in the worktree — the live production shape.
    Mirrors job-cannon's ``15dacbb6``, which deleted ``.devin/prompts/`` from
    main while the devin shim keeps rewriting those files in every worktree
    forked before that commit. The local modification is restored from HEAD
    and the merge retried once, succeeding.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Common ancestor: both branches start out tracking the scaffolding file.
    (repo_root / ".devin" / "prompts").mkdir(parents=True)
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "base scaffolding v0\n", encoding="utf-8"
    )
    _git(repo_root, "add", ".devin/prompts/worker.md")
    _git(repo_root, "commit", "-m", "add scaffolding to ancestor")

    _git(repo_root, "checkout", "-b", "feature")
    (repo_root / "work.txt").write_text("worker output\n", encoding="utf-8")
    _git(repo_root, "add", "work.txt")
    _git(repo_root, "commit", "-m", "feature work")

    _git(repo_root, "checkout", "main")
    # job-cannon 15dacbb6: the base deletes the scaffolding path outright.
    _git(repo_root, "rm", ".devin/prompts/worker.md")
    _git(repo_root, "commit", "-m", "delete scaffolding from main")

    _git(repo_root, "checkout", "feature")
    # The devin shim rewrites the file in place, uncommitted — a locally
    # modified TRACKED file, not an untracked one.
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "shim-rewritten content\n", encoding="utf-8"
    )

    result = _merge_update_rework_branch(
        repo_root,
        repo_root,
        "feature",
        "main",
        injected_paths=(".devin/prompts/worker.md",),
    )

    assert result is None
    merge_head_check = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", "MERGE_HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert merge_head_check.returncode != 0
    # The base's deletion won the merge; the shim's local edit is gone.
    assert not (repo_root / ".devin" / "prompts" / "worker.md").exists()
    # An unrelated worker-authored tracked file must survive untouched.
    assert (repo_root / "work.txt").read_text(encoding="utf-8") == "worker output\n"


def test_modified_paths_overwritten_by_ref_uses_merge_base_not_ref_directly(
    tmp_path: Path,
) -> None:
    """The incoming side must be computed against the merge base, not against
    ``ref`` directly. A path the *branch* changed since the ancestor, which
    the base left untouched, is not something merging the base would touch —
    including it would restore a branch's own local edit for no reason.

    Discriminating: if the implementation diffed ``HEAD..ref`` instead of
    ``merge_base..ref`` for the incoming side, ``branch_only.txt`` would
    appear in the incoming set too (HEAD already carries the branch's own
    committed change to it, which differs from ``ref``), and the result would
    wrongly include both paths instead of just the one the base changed.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    (repo_root / "base_changes.txt").write_text("ancestor v0\n", encoding="utf-8")
    (repo_root / "branch_only.txt").write_text("ancestor v0\n", encoding="utf-8")
    _git(repo_root, "add", "base_changes.txt", "branch_only.txt")
    _git(repo_root, "commit", "-m", "add ancestor files")

    _git(repo_root, "checkout", "-b", "feature")
    (repo_root / "branch_only.txt").write_text("feature committed v1\n", encoding="utf-8")
    _git(repo_root, "add", "branch_only.txt")
    _git(repo_root, "commit", "-m", "feature-only change")

    _git(repo_root, "checkout", "main")
    (repo_root / "base_changes.txt").write_text("main committed v1\n", encoding="utf-8")
    _git(repo_root, "add", "base_changes.txt")
    _git(repo_root, "commit", "-m", "base-only change")

    _git(repo_root, "checkout", "feature")
    # Locally modify BOTH tracked files, uncommitted, so both are eligible on
    # the "dirty" side — the discriminator is purely the incoming side.
    (repo_root / "base_changes.txt").write_text("locally edited\n", encoding="utf-8")
    (repo_root / "branch_only.txt").write_text("locally edited\n", encoding="utf-8")

    result = _modified_paths_overwritten_by_ref(repo_root, "main")

    assert result == ("base_changes.txt",)
    assert "branch_only.txt" not in result


def test_modified_paths_overwritten_by_ref_excludes_staged_addition(
    tmp_path: Path,
) -> None:
    """A staged ADDITION — a path in the index but absent from HEAD — must not
    appear in the result even when the base ref independently adds a file at
    the same path: ``git checkout HEAD --`` cannot restore a path HEAD never
    had, so including it would hand the caller a repair it cannot perform.

    A genuine modification (present in both HEAD and the worktree) is the
    positive control proving the function is not just returning empty.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    _seed_staged_addition_bubble(repo_root)
    (repo_root / "tracked.txt").write_text("locally edited\n", encoding="utf-8")
    # A staged ADDITION: in the index, not in HEAD. Same path name as
    # something the base independently added, so an unfiltered diff would
    # intersect them.
    (repo_root / "new_file.txt").write_text("locally staged addition\n", encoding="utf-8")
    _git(repo_root, "add", "new_file.txt")

    result = _modified_paths_overwritten_by_ref(repo_root, "main")

    assert result == ("tracked.txt",)
    assert "new_file.txt" not in result


def test_merge_update_rework_branch_undeclared_modified_blocker_raises_pre_merge(
    tmp_path: Path,
) -> None:
    """A locally-modified tracked file OUTSIDE the declared scaffolding blocks
    the merge exactly like an undeclared untracked collision does: escalate
    as ``stage="pre_merge"`` with the file left as modified on disk.

    A *declared* scaffolding modification present in the SAME worktree must
    also be left untouched. Both blockers here land in the same (modified)
    class, so this alone does not distinguish a union gate from a gate that
    decides declared/undeclared independently per class — see
    ``test_merge_update_rework_branch_undeclared_untracked_blocker_skips_declared_modified``
    below for the cross-class variant that does.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    (repo_root / "shared.txt").write_text("ancestor v0\n", encoding="utf-8")
    (repo_root / ".devin" / "prompts").mkdir(parents=True)
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "ancestor scaffolding v0\n", encoding="utf-8"
    )
    _git(repo_root, "add", "shared.txt", ".devin/prompts/worker.md")
    _git(repo_root, "commit", "-m", "add ancestor files")

    _git(repo_root, "checkout", "-b", "feature")
    (repo_root / "work.txt").write_text("worker output\n", encoding="utf-8")
    _git(repo_root, "add", "work.txt")
    _git(repo_root, "commit", "-m", "feature work")

    _git(repo_root, "checkout", "main")
    (repo_root / "shared.txt").write_text("main committed v1\n", encoding="utf-8")
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "main scaffolding v1\n", encoding="utf-8"
    )
    _git(repo_root, "add", "shared.txt", ".devin/prompts/worker.md")
    _git(repo_root, "commit", "-m", "base changes both files")

    _git(repo_root, "checkout", "feature")
    # Undeclared: a worker-authored file that happens to collide with a
    # base-tracked path.
    (repo_root / "shared.txt").write_text("worker's own copy\n", encoding="utf-8")
    # Declared scaffolding, also modified — repairable on its own, but must
    # not be repaired here because of the undeclared blocker above.
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "shim-rewritten content\n", encoding="utf-8"
    )

    with pytest.raises(ReworkBranchConflictError) as exc_info:
        _merge_update_rework_branch(
            repo_root,
            repo_root,
            "feature",
            "main",
            injected_paths=(".devin/prompts/worker.md",),
        )

    assert exc_info.value.stage == "pre_merge"
    assert "shared.txt" in exc_info.value.conflicted_paths
    # Nothing declared was touched: the undeclared blocker refuses repair of
    # BOTH classes, not just its own.
    assert (repo_root / "shared.txt").read_text(encoding="utf-8") == "worker's own copy\n"
    devin_prompt = repo_root / ".devin" / "prompts" / "worker.md"
    assert devin_prompt.read_text(encoding="utf-8") == "shim-rewritten content\n"


def test_merge_update_rework_branch_undeclared_untracked_blocker_skips_declared_modified(
    tmp_path: Path,
) -> None:
    """Cross-class variant of the union-gate property above: the undeclared
    blocker is in the UNTRACKED class and the declared, individually
    repairable blocker is in the MODIFIED class.

    A gate that decides declared/undeclared per class independently (rather
    than over the union of both classes) would find the modified class free
    of undeclared members and repair it anyway — silently discarding the
    worker's locally modified scaffolding file moments before the merge
    still escalates on the untracked blocker. The same-class test above
    cannot catch this divergence: with both blockers in one class, a
    per-class gate and the union gate make the identical decision.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    (repo_root / ".devin" / "prompts").mkdir(parents=True)
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "ancestor scaffolding v0\n", encoding="utf-8"
    )
    _git(repo_root, "add", ".devin/prompts/worker.md")
    _git(repo_root, "commit", "-m", "add ancestor scaffolding")

    _git(repo_root, "checkout", "-b", "feature")

    _git(repo_root, "checkout", "main")
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "main scaffolding v1\n", encoding="utf-8"
    )
    _git(repo_root, "add", ".devin/prompts/worker.md")
    # A path main tracks that feature never had — a same-named local file in
    # the feature worktree is therefore untracked, not modified.
    (repo_root / "shared.txt").write_text("main version\n", encoding="utf-8")
    _git(repo_root, "add", "shared.txt")
    _git(repo_root, "commit", "-m", "base changes scaffolding and adds shared.txt")

    _git(repo_root, "checkout", "feature")
    # Declared, modified-class blocker — repairable on its own.
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "shim-rewritten content\n", encoding="utf-8"
    )
    # Undeclared, untracked-class blocker.
    (repo_root / "shared.txt").write_text("worker's own copy\n", encoding="utf-8")

    with pytest.raises(ReworkBranchConflictError) as exc_info:
        _merge_update_rework_branch(
            repo_root,
            repo_root,
            "feature",
            "main",
            injected_paths=(".devin/prompts/worker.md",),
        )

    assert exc_info.value.stage == "pre_merge"
    assert "shared.txt" in exc_info.value.conflicted_paths
    # The declared, modified-class file must be left exactly as the worker
    # left it: an undeclared blocker in the OTHER class still refuses this
    # class's repair under the union gate.
    devin_prompt = repo_root / ".devin" / "prompts" / "worker.md"
    assert devin_prompt.read_text(encoding="utf-8") == "shim-rewritten content\n"
    assert (repo_root / "shared.txt").read_text(encoding="utf-8") == "worker's own copy\n"


def test_restore_declared_scaffolding_modifications_discards_staged_edit(
    tmp_path: Path,
) -> None:
    """A locally modified declared-scaffolding file that has also been staged
    (``git add``ed) must be fully discarded and the retried merge must
    succeed. This distinguishes ``git checkout HEAD --`` from bare
    ``git checkout --``: the latter restores the working tree from the INDEX,
    which still holds the staged edit, so the retry would fail on the same
    path a second time.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    (repo_root / ".devin" / "prompts").mkdir(parents=True)
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "ancestor scaffolding v0\n", encoding="utf-8"
    )
    _git(repo_root, "add", ".devin/prompts/worker.md")
    _git(repo_root, "commit", "-m", "add ancestor scaffolding")

    _git(repo_root, "checkout", "-b", "feature")

    _git(repo_root, "checkout", "main")
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "main committed v1\n", encoding="utf-8"
    )
    _git(repo_root, "add", ".devin/prompts/worker.md")
    _git(repo_root, "commit", "-m", "base changes scaffolding")

    _git(repo_root, "checkout", "feature")
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "feature locally staged edit\n", encoding="utf-8"
    )
    _git(repo_root, "add", ".devin/prompts/worker.md")  # staged, not committed

    result = _merge_update_rework_branch(
        repo_root,
        repo_root,
        "feature",
        "main",
        injected_paths=(".devin/prompts/worker.md",),
    )

    assert result is None
    merge_head_check = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", "MERGE_HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert merge_head_check.returncode != 0
    devin_prompt = repo_root / ".devin" / "prompts" / "worker.md"
    assert devin_prompt.read_text(encoding="utf-8") == "main committed v1\n"


def test_restore_declared_scaffolding_modifications_refuses_venv_paths(
    tmp_path: Path,
) -> None:
    """``.venv`` must never be restored by the pre-merge scaffolding repair,
    even if it were somehow declared — mirrors
    ``test_clear_declared_scaffolding_collisions_refuses_venv_paths`` for the
    restore (modified) side rather than the clear (untracked) side.

    Includes a positive control on an ordinary (non-``.venv``) declared,
    genuinely modified path in the same repo: without it, "returns False" for
    the ``.venv`` case would be equally consistent with the function being
    unconditionally broken rather than specifically refusing ``.venv``.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # The marker is committed, then locally modified, so a `git checkout
    # HEAD -- .venv/pyvenv.cfg` WOULD succeed and restore the committed
    # content if the .venv guard were dropped. An untracked-only marker
    # (as in an earlier version of this test) would make `restored is
    # False` a foregone conclusion regardless of the guard, since the
    # checkout would fail on a bad pathspec either way — that shape cannot
    # fail and was not actually exercising the refusal.
    venv_dir = repo_root / ".venv"
    venv_dir.mkdir()
    marker = venv_dir / "pyvenv.cfg"
    marker.write_text("home = committed\n", encoding="utf-8")
    _git(repo_root, "add", ".venv/pyvenv.cfg")
    _git(repo_root, "commit", "-m", "add tracked venv marker")
    marker.write_text("home = locally modified\n", encoding="utf-8")

    restored = _restore_declared_scaffolding_modifications(
        repo_root,
        (".venv/pyvenv.cfg",),
        (".venv",),
        (),
    )

    assert restored is False
    assert marker.read_text(encoding="utf-8") == "home = locally modified\n"

    # Positive control: an ordinary declared, genuinely modified tracked file
    # IS restored. This proves False above means "refused", not "the
    # function can never return True."
    (repo_root / "prompts").mkdir()
    ordinary = repo_root / "prompts" / "PROMPT.md"
    ordinary.write_text("committed version\n", encoding="utf-8")
    _git(repo_root, "add", "prompts/PROMPT.md")
    _git(repo_root, "commit", "-m", "add ordinary scaffolding")
    ordinary.write_text("locally modified\n", encoding="utf-8")

    restored_ordinary = _restore_declared_scaffolding_modifications(
        repo_root,
        ("prompts/PROMPT.md",),
        ("prompts/PROMPT.md",),
        (),
    )

    assert restored_ordinary is True
    assert ordinary.read_text(encoding="utf-8") == "committed version\n"


def test_eligible_for_scaffolding_repair_gate(tmp_path: Path) -> None:
    """The shared safety gate every scaffolding repair passes through, tested
    directly rather than only through its two callers' side effects: empty
    input, an undeclared path, a ``.venv`` path (even when declared), and a
    path that escapes the worktree via ``..`` must all refuse. An ordinary
    declared, contained path is the positive control proving the gate can
    also say yes.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    assert _eligible_for_scaffolding_repair(repo_root, (), ("prompts/PROMPT.md",), ()) is False
    assert (
        _eligible_for_scaffolding_repair(
            repo_root, ("undeclared.txt",), ("prompts/PROMPT.md",), ()
        )
        is False
    )
    assert (
        _eligible_for_scaffolding_repair(repo_root, (".venv/pyvenv.cfg",), (".venv",), ()) is False
    )
    assert (
        _eligible_for_scaffolding_repair(repo_root, ("../outside.txt",), ("../outside.txt",), ())
        is False
    )
    assert (
        _eligible_for_scaffolding_repair(
            repo_root, ("prompts/PROMPT.md",), ("prompts/PROMPT.md",), ()
        )
        is True
    )


def test_repair_declared_scaffolding_blockers_only_modified_class(tmp_path: Path) -> None:
    """A worktree blocked by only the modified class must still be repaired —
    an empty ``untracked_blocking`` is skipped rather than treated as a
    failure, not required to also be non-empty.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    (repo_root / "prompts").mkdir()
    scaffolding = repo_root / "prompts" / "PROMPT.md"
    scaffolding.write_text("committed version\n", encoding="utf-8")
    _git(repo_root, "add", "prompts/PROMPT.md")
    _git(repo_root, "commit", "-m", "add scaffolding")
    scaffolding.write_text("locally modified\n", encoding="utf-8")

    repaired = _repair_declared_scaffolding_blockers(
        repo_root,
        (),
        ("prompts/PROMPT.md",),
        ("prompts/PROMPT.md",),
        (),
    )

    assert repaired is True
    assert scaffolding.read_text(encoding="utf-8") == "committed version\n"


def test_repair_declared_scaffolding_blockers_only_untracked_class(tmp_path: Path) -> None:
    """A worktree blocked by only the untracked class must still be repaired —
    an empty ``modified_blocking`` is skipped rather than treated as a
    failure, not required to also be non-empty.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    (repo_root / "prompts").mkdir()
    shadow = repo_root / "prompts" / "PROMPT.md"
    shadow.write_text("untracked shadow copy\n", encoding="utf-8")

    repaired = _repair_declared_scaffolding_blockers(
        repo_root,
        ("prompts/PROMPT.md",),
        (),
        ("prompts/PROMPT.md",),
        (),
    )

    assert repaired is True
    assert not shadow.exists()


def test_repair_declared_scaffolding_blockers_both_empty_returns_false(
    tmp_path: Path,
) -> None:
    """Neither class populated means there was nothing to repair — this must
    be a refusal, not a vacuous success (which would let the caller retry a
    merge no repair actually touched).
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    repaired = _repair_declared_scaffolding_blockers(repo_root, (), (), (), ())

    assert repaired is False
