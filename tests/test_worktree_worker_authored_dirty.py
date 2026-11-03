"""``_worker_authored_dirty``: which uncommitted paths count as worker output versus
orchestrator-injected or launcher-owned scaffolding."""

from __future__ import annotations

from pathlib import Path

from _worktree_fixtures import _git, _init_repo
from charlie_work import worktree as worktree_module
from charlie_work.worktree import _worker_authored_dirty


def test_worker_authored_dirty_ignores_worker_outcome_marker(tmp_path: Path) -> None:
    """Issue #989: ``.worker-outcome.json`` is protocol scaffolding, not worker work.

    It is written on the one path where the worker pushed a branch but could not open
    a PR, so treating it as worker-authored made every such worktree permanently
    ineligible for ``clean_worktrees`` -- including after the salvage path opened the
    PR and it merged.

    Goes through ``DispatchConfig``'s real ``injected_paths`` rather than passing the
    filename in by hand: the property under test is that the shipped configuration
    excludes it, which a hand-passed tuple would assert nothing about.
    """
    from charlie_work.config import DispatchConfig, WORKER_OUTCOME_FILENAME

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / WORKER_OUTCOME_FILENAME).write_text(
        '{"push_succeeded": true, "pr_created": false, "error": "gh unauthenticated"}',
        encoding="utf-8",
    )

    assert _worker_authored_dirty(repo_root, DispatchConfig().injected_paths) is False


def test_worker_authored_dirty_still_flags_work_beside_worker_outcome_marker(
    tmp_path: Path,
) -> None:
    """Issue #989 negative control: excluding the marker must not excuse its neighbours.

    Without this, a fix that over-broadly reported every worktree clean would pass the
    test above and silently strand real work.
    """
    from charlie_work.config import DispatchConfig, WORKER_OUTCOME_FILENAME

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / WORKER_OUTCOME_FILENAME).write_text(
        '{"push_succeeded": true, "pr_created": false, "error": "gh unauthenticated"}',
        encoding="utf-8",
    )
    (repo_root / "worker_authored.py").write_text("# real work\n", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, DispatchConfig().injected_paths) is True


def test_worker_outcome_filename_matches_the_prompt_contract() -> None:
    """Issue #989: the constant and the prompt that tells workers what to write must agree.

    ``push_pr_outcome.md`` names the file literally, so a rename of the constant that
    missed the prompt would leave workers writing a name nothing reads -- and the
    failure is silent, because a missing outcome file is indistinguishable from a
    worker that never hit the unauthenticated path.
    """
    from charlie_work.config import WORKER_OUTCOME_FILENAME

    prompt = (
        Path(worktree_module.__file__).parent
        / "prompts"
        / "worker_sections"
        / "push_pr_outcome.md"
    ).read_text(encoding="utf-8")

    assert WORKER_OUTCOME_FILENAME in prompt


def test_worker_outcome_filename_is_re_exported_from_worktree() -> None:
    """Issue #989: the constant moved to ``config`` to break a circular import.

    ``worktree.WORKER_OUTCOME_FILENAME`` stays valid for existing importers, and must
    remain the same object -- two independent string literals would drift apart
    silently, which is the bug this move exists to prevent.
    """
    from charlie_work.config import WORKER_OUTCOME_FILENAME as CONFIG_NAME

    assert worktree_module.WORKER_OUTCOME_FILENAME is CONFIG_NAME


def test_worker_authored_dirty_ignores_custom_override_path(tmp_path: Path) -> None:
    """Issue #381: a custom injected_paths override excludes the configured path."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    injected = repo_root / ".devin" / "prompts" / "worker.md"
    injected.parent.mkdir(parents=True)
    injected.write_text("injected prompt", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, (".devin/prompts/worker.md",)) is False


def test_worker_authored_dirty_normalizes_backslash_in_override(tmp_path: Path) -> None:
    """Issue #381: a Windows-style backslash override matches git's forward-slash path."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    injected = repo_root / ".devin" / "prompts" / "worker.md"
    injected.parent.mkdir(parents=True)
    injected.write_text("injected prompt", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, (".devin\\prompts\\worker.md",)) is False


def test_worker_authored_dirty_detects_sibling_in_collapsed_untracked_dir(
    tmp_path: Path,
) -> None:
    """Issue #381 follow-up, now structural: ``--untracked-files=all`` means
    git never collapses a wholly-untracked directory into a single ``??
    dir/`` line in the first place — every file, including a worker-authored
    sibling living next to a nested injected path, gets its own record. There
    is no collapsed line left to special-case or re-probe.

    Uses a non-launcher-owned directory (``.custom/``) because ``.devin/`` is
    now entirely launcher-owned (issue #1391) — a sibling under ``.devin/``
    is correctly ignored, so it cannot serve as the "should be detected"
    case here.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    injected = repo_root / ".custom" / "prompts" / "worker.md"
    injected.parent.mkdir(parents=True)
    injected.write_text("injected prompt", encoding="utf-8")
    sibling = repo_root / ".custom" / "worker-output.txt"
    sibling.write_text("real worker output", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, (".custom/prompts/worker.md",)) is True


def test_worker_authored_dirty_untracked_dir_with_only_injected_stays_clean(
    tmp_path: Path,
) -> None:
    """Inverse control: a directory containing ONLY the injected path (no
    siblings) must still be excused, preserving issue #381's original
    carve-out under full per-file enumeration.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    injected = repo_root / ".devin" / "prompts" / "worker.md"
    injected.parent.mkdir(parents=True)
    injected.write_text("injected prompt", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, (".devin/prompts/worker.md",)) is False


def test_worker_authored_dirty_tracked_leading_dot_path_modified_in_place(
    tmp_path: Path,
) -> None:
    """Issue #381 bug 1 (v1 rework history): the old line-based parser called
    ``.strip()`` on each porcelain line before a fixed-column ``line[3:]``
    slice, which corrupted the leading-space status column of an unstaged
    tracked modification (e.g. ``" M .orchestrator-prompt.md"``) — the strip
    shifted the string left by one character and dropped the leading dot.
    ``--porcelain=v2 -z`` has no line-based whitespace to strip in the first
    place: fields are parsed by splitting the record on single spaces with a
    bounded maxsplit, never by slicing a stripped string.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    prompt = repo_root / ".orchestrator-prompt.md"
    prompt.write_text("original prompt", encoding="utf-8")
    _git(repo_root, "add", str(prompt))
    _git(repo_root, "commit", "-m", "track prompt")
    # Modify in place without staging: unstaged tracked change, XY = ".M".
    prompt.write_text("rewritten prompt", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, (".orchestrator-prompt.md",)) is False

    # Control: the same leading-dot tracked-modification shape, but for a path
    # NOT in injected_paths, must still be detected as dirty.
    other = repo_root / ".other-tracked.md"
    other.write_text("original", encoding="utf-8")
    _git(repo_root, "add", str(other))
    _git(repo_root, "commit", "-m", "track other")
    other.write_text("changed", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, (".orchestrator-prompt.md",)) is True


def test_worker_authored_dirty_directory_injected_path_scoped_to_its_own_tree(
    tmp_path: Path,
) -> None:
    """Issue #381 bug 4 (v1 rework history): when an injected_paths entry
    names a whole directory rather than a specific file (a documented
    supported convention — see ``DispatchConfig.injected_paths``), the old
    line-based parser matched it via an exact-match fast path that ran
    *before*, and bypassed, the collapse-safety re-probe used for the
    nested-file case (bug 3) — an inconsistency between two code paths meant
    to answer the same question. ``--untracked-files=all`` removes the
    re-probe machinery (and the fast-path/re-probe distinction) entirely:
    every file, whether it matches an injected file or an injected directory,
    is matched individually by the exact same normalized-path predicate.

    A directory-level injected_paths entry legitimately excuses everything
    under it (that is the documented, intentional trade-off of naming a
    whole directory) — but the exclusion must stay scoped to that directory's
    own subtree and not leak to unrelated content, including a directory that
    merely shares a string prefix.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    injected = repo_root / ".devin" / "prompts" / "worker.md"
    injected.parent.mkdir(parents=True)
    injected.write_text("injected prompt", encoding="utf-8")
    sibling = repo_root / ".devin" / "worker-output.txt"
    sibling.write_text("real worker output inside the excluded tree", encoding="utf-8")

    # Naming the directory itself excuses everything inside it, by design.
    assert _worker_authored_dirty(repo_root, (".devin",)) is False

    # An unrelated untracked file OUTSIDE the named directory must still be
    # detected — the exclusion does not leak beyond its own subtree.
    unrelated = repo_root / "worker-result.txt"
    unrelated.write_text("unrelated worker output", encoding="utf-8")
    assert _worker_authored_dirty(repo_root, (".devin",)) is True


def test_worker_authored_dirty_directory_prefix_does_not_collide(tmp_path: Path) -> None:
    """A directory that merely shares a string prefix with a configured
    injected directory (e.g. ``.devin-cache`` vs ``.devin``) must not match —
    path-segment comparison via ``PurePosixPath.parents``, not substring
    matching, is what makes this safe.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    cache_dir = repo_root / ".devin-cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "marker.txt").write_text("not injected", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, (".devin",)) is True


def test_worker_authored_dirty_renamed_tracked_file_not_injected(tmp_path: Path) -> None:
    """A rename/copy porcelain=v2 record (tag ``2``) carries an extra
    NUL-delimited ``origPath`` field after the current path. Parsing must
    consume that extra field so it isn't mistaken for the next record's tag,
    and the CURRENT (renamed-to) path is what gets matched against
    injected_paths — mirroring the old parser's ``" -> "`` right-hand-side
    convention.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    tracked = repo_root / "tracked.txt"
    tracked.write_text("content\n", encoding="utf-8")
    _git(repo_root, "add", "tracked.txt")
    _git(repo_root, "commit", "-m", "add tracked")
    _git(repo_root, "mv", "tracked.txt", "renamed.txt")
    # A subsequent untracked file proves the rename record's extra origPath
    # field was correctly consumed rather than corrupting the next record.
    (repo_root / "new-worker-file.txt").write_text("also here", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, ()) is True

    # A rename INTO an injected path is excused by the current-path match.
    _git(repo_root, "reset", "--hard")
    (repo_root / "new-worker-file.txt").unlink(missing_ok=True)
    _git(repo_root, "mv", "tracked.txt", "prompt-renamed.md")
    assert _worker_authored_dirty(repo_root, ("prompt-renamed.md",)) is False


def test_worker_authored_dirty_excludes_materialize_dirs_surface(
    tmp_path: Path,
) -> None:
    """Issue #471: tracked modifications confined to the configured
    ``materialize_dirs`` surface must not count as worker-authored dirt,
    even when the assume-unchanged bit is not set.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    prompt = repo_root / ".devin" / "prompts" / "worker.md"
    prompt.parent.mkdir(parents=True, exist_ok=True)
    prompt.write_text("original prompt\n", encoding="utf-8")
    _git(repo_root, "add", str(prompt))
    _git(repo_root, "commit", "-m", "track prompt template")

    # Simulate an external launch shim rewriting the tracked prompt in place
    # without the assume-unchanged bit set.
    prompt.write_text("per-dispatch prompt\n", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, (), (".devin",)) is False


def test_worker_authored_dirty_detects_changes_outside_materialize_and_injected(
    tmp_path: Path,
) -> None:
    """Issue #471: a modification outside both ``injected_paths`` and
    ``materialize_dirs`` must still trip the worktree-safety guard.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    (repo_root / "worker-output.txt").write_text("worker result\n", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, (), (".devin",)) is True


# --- issue #1391: launcher-owned shim dirt is not worker output ---------------


def test_worker_authored_dirty_ignores_devin_shim_residue(tmp_path: Path) -> None:
    """Issue #1391: untracked ``.devin/`` launcher residue (AGENTS.md,
    hooks.v1.json, hooks/*.py, skills/, worker.md, orchestrator.md) is shim
    dirt, not worker output. The dirty check must ignore it so a worktree
    whose only dirt is launcher residue is not refused re-arm."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    # Simulate the full set of shim residue observed in the field.
    devin = repo_root / ".devin"
    devin.mkdir(parents=True)
    (devin / "AGENTS.md").write_text("agents\n", encoding="utf-8")
    (devin / "worker.md").write_text("worker\n", encoding="utf-8")
    (devin / "orchestrator.md").write_text("orch\n", encoding="utf-8")
    (devin / "hooks.v1.json").write_text("{}", encoding="utf-8")
    hooks_dir = devin / "hooks"
    hooks_dir.mkdir(parents=True)
    (hooks_dir / "pre_tool.py").write_text("# hook\n", encoding="utf-8")
    skills_dir = devin / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "foo.py").write_text("# skill\n", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, ()) is False


def test_worker_authored_dirty_ignores_devin_prompts_modifications(
    tmp_path: Path,
) -> None:
    """Issue #1391: modified ``.devin/prompts/rework.md`` / ``worker.md`` are
    shim rewrites, not worker output. Even when tracked, modifications under
    ``.devin/`` must be ignored by the launcher-owned matcher (independent of
    ``injected_paths`` / ``materialize_dirs`` config)."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    prompts_dir = repo_root / ".devin" / "prompts"
    prompts_dir.mkdir(parents=True)
    for name in ("worker.md", "rework.md"):
        p = prompts_dir / name
        p.write_text("original\n", encoding="utf-8")
        _git(repo_root, "add", str(p))
    _git(repo_root, "commit", "-m", "track devin prompts")
    # Simulate the shim rewriting them.
    (prompts_dir / "worker.md").write_text("rewritten\n", encoding="utf-8")
    (prompts_dir / "rework.md").write_text("rewritten\n", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, ()) is False


def test_worker_authored_dirty_ignores_pr_body_scratch_files(tmp_path: Path) -> None:
    """Issue #1391: PR body scratch files (``PR_BODY.md``, ``PR_BODY_42.md``,
    ``.worker-pr-body.md``, ``_pr_body.md``) are launcher/protocol residue,
    not worker output. The dirty check must ignore all variants."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    for name in ("PR_BODY.md", "PR_BODY_42.md", ".worker-pr-body.md", "_pr_body.md"):
        (repo_root / name).write_text("# PR body draft\n", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, ()) is False


def test_worker_authored_dirty_ignores_git_worktree_dir(tmp_path: Path) -> None:
    """Issue #1391: untracked ``.git_worktree_dir/`` is launcher residue."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    wt_dir = repo_root / ".git_worktree_dir"
    wt_dir.mkdir(parents=True)
    (wt_dir / "cache.json").write_text("{}", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, ()) is False


def test_worker_authored_dirty_ignores_charlie_writer_deletion(tmp_path: Path) -> None:
    """Issue #1391: a deleted ``.charlie-writer.json`` marker is protocol
    residue, not worker output. The writer marker is already in the default
    ``injected_paths``, so a deletion is excluded by the declared-scaffolding
    matcher — this test pins that the default config covers the deletion
    case (not just creation/modification)."""
    from charlie_work.config import DispatchConfig

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    marker = repo_root / ".charlie-writer.json"
    marker.write_text('{"pid": 1}\n', encoding="utf-8")
    _git(repo_root, "add", str(marker))
    _git(repo_root, "commit", "-m", "track writer marker")
    # Delete it (simulating a worker or shim removing the marker).
    marker.unlink()

    assert _worker_authored_dirty(repo_root, DispatchConfig().injected_paths) is False


def test_worker_authored_dirty_still_flags_real_work_alongside_shim_dirt(
    tmp_path: Path,
) -> None:
    """Issue #1391 negative control: ignoring launcher residue must NOT excuse
    genuine worker-authored modifications mixed in with the junk. This is the
    jc #1514 scenario — two real modified source files alongside shim dirt
    that were lost when the worktree was force-removed."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    # Shim residue.
    devin = repo_root / ".devin"
    devin.mkdir(parents=True)
    (devin / "AGENTS.md").write_text("agents\n", encoding="utf-8")
    (repo_root / "PR_BODY.md").write_text("# draft\n", encoding="utf-8")
    # Real worker output — a tracked source file modification.
    src = repo_root / "src" / "app.py"
    src.parent.mkdir(parents=True)
    src.write_text("original\n", encoding="utf-8")
    _git(repo_root, "add", str(src))
    _git(repo_root, "commit", "-m", "add app")
    src.write_text("modified by worker\n", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, ()) is True


def test_worker_authored_dirty_devin_prefix_does_not_collide(tmp_path: Path) -> None:
    """Issue #1391: a directory that merely shares a string prefix with a
    launcher-owned dir (``.devin-cache`` vs ``.devin``) must not match —
    path-segment comparison, not substring matching."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    cache_dir = repo_root / ".devin-cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "marker.txt").write_text("not launcher\n", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, ()) is True
