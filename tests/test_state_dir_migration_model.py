"""Value objects and pure path helpers for ``charlie_work.state_migration``.

Split out of ``tests/test_state_dir_migration.py`` (issue #1566, Track-1):
the frozen-dataclass contracts (``MigrationChild`` / ``MigrationPlan`` /
``MigrationOutcome`` / ``StateRewriteResult``) and the pure path-matching
primitives (``_normalize_path_key`` / ``_is_equal_or_nested`` /
``_relative_parts``) that the separator/case defect and the nested-
worktree remediation target depend on. No filesystem or git access.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from _state_dir_migration_fixtures import DST_ROOT, SRC_ROOT
from charlie_work.state_migration import (
    MigrationChild,
    MigrationOutcome,
    MigrationPlan,
    StateRewriteResult,
    _is_equal_or_nested,
    _normalize_path_key,
    _relative_parts,
)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


def test_value_objects_are_frozen() -> None:
    """Migration value objects follow the project's frozen-dataclass rule."""
    child = MigrationChild(
        name="x",
        src_path=SRC_ROOT / "x",
        dst_path=DST_ROOT / "x",
        disposition="move",
    )
    with pytest.raises(Exception):
        child.disposition = "blocked"  # type: ignore[misc]

    plan = MigrationPlan(src_root=SRC_ROOT, dst_root=DST_ROOT, children=())
    with pytest.raises(Exception):
        plan.ok = False  # type: ignore[misc]


def test_migration_child_movable_and_blocked_properties() -> None:
    moved = MigrationChild(
        name="a", src_path=SRC_ROOT / "a", dst_path=DST_ROOT / "a", disposition="move"
    )
    blocked = MigrationChild(
        name="b", src_path=SRC_ROOT / "b", dst_path=DST_ROOT / "b", disposition="blocked"
    )
    assert moved.movable is True
    assert moved.blocked is False
    assert blocked.movable is False
    assert blocked.blocked is True


def test_migration_plan_movable_and_blocked_filter_children() -> None:
    moved = MigrationChild(
        name="a", src_path=SRC_ROOT / "a", dst_path=DST_ROOT / "a", disposition="move"
    )
    blocked = MigrationChild(
        name="b", src_path=SRC_ROOT / "b", dst_path=DST_ROOT / "b", disposition="blocked"
    )
    plan = MigrationPlan(src_root=SRC_ROOT, dst_root=DST_ROOT, children=(moved, blocked))
    assert plan.movable == (moved,)
    assert plan.blocked == (blocked,)


def test_migration_outcome_is_frozen() -> None:
    """MigrationOutcome follows the project's frozen-dataclass rule."""
    outcome = MigrationOutcome(ok=True, moved=("issues",))
    with pytest.raises(Exception):
        outcome.ok = False  # type: ignore[misc]


def test_migration_outcome_has_rewritten_paths_field_default_zero() -> None:
    """The new field exists and defaults to 0 when not specified."""
    outcome = MigrationOutcome(ok=True, moved=("issues",))
    assert outcome.rewritten_paths == 0


def test_state_rewrite_result_is_frozen() -> None:
    """StateRewriteResult follows the project's frozen-dataclass rule."""
    result = StateRewriteResult(ok=True, rewritten=3)
    with pytest.raises(Exception):
        result.ok = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _normalize_path_key / _is_equal_or_nested -- the separator/case defect
# ---------------------------------------------------------------------------


def test_normalize_path_key_folds_separator_and_case() -> None:
    """A forward-slash git-style path and a backslash disk-style path for the
    exact same location, differing in case too, must produce the same key.

    ``PurePosixPath`` is used for the "git" side specifically because on
    Windows, ``pathlib.Path("C:/x/y")`` already silently renders as
    backslashes when stringified -- that native normalization would make this
    test pass even if ``_normalize_path_key`` did nothing, and would not
    actually pin the historical defect. ``PurePosixPath`` never substitutes
    the separator and is never case-folded, so equality here is proof the
    module's own normalization is doing the work, independent of host OS.
    """
    git_style = PurePosixPath("C:/Users/operator/repos/Job-Cannon/.claude/Worktrees/FOO")
    disk_style = Path(r"C:\Users\OPERATOR\repos\job-cannon\.claude\worktrees\foo")

    assert _normalize_path_key(git_style) == _normalize_path_key(disk_style)  # type: ignore[arg-type]


def test_normalize_path_key_differs_for_genuinely_different_paths() -> None:
    assert _normalize_path_key(Path("C:/a/b")) != _normalize_path_key(Path("C:/a/c"))


def test_is_equal_or_nested_true_for_forward_slash_git_path_vs_backslash_disk_path() -> None:
    """The regression that made a real probe report 74 registered worktrees as
    orphaned: a git-porcelain path (forward slash) for a worktree must be
    recognized as equal to the same location expressed with backslashes.
    """
    registered = PurePosixPath("C:/repos/job-cannon/.var/charlie-work/worktrees/agent-abc123")
    on_disk = Path(r"C:\repos\job-cannon\.var\charlie-work\worktrees\agent-abc123")

    assert _is_equal_or_nested(registered, on_disk)  # type: ignore[arg-type]


def test_is_equal_or_nested_true_for_exact_match() -> None:
    same = Path("C:/state/dispatches")
    assert _is_equal_or_nested(same, Path("C:/state/dispatches"))


def test_is_equal_or_nested_true_for_deeply_nested_child() -> None:
    outer = Path("C:/state/dispatches")
    inner = Path("C:/state/dispatches/reviews/pr-1384")
    assert _is_equal_or_nested(inner, outer)


def test_is_equal_or_nested_false_for_sibling_with_shared_prefix() -> None:
    """A trailing separator must be part of the prefix check, or a sibling
    like ``worktrees-old`` would be wrongly treated as nested inside
    ``worktrees`` merely because the strings share a prefix.
    """
    outer = Path("C:/state/worktrees")
    sibling = Path("C:/state/worktrees-old/pr-1")
    assert not _is_equal_or_nested(sibling, outer)


def test_is_equal_or_nested_false_for_unrelated_path() -> None:
    assert not _is_equal_or_nested(Path("C:/state/issues"), Path("C:/state/dispatches"))


# ---------------------------------------------------------------------------
# _relative_parts -- remediation-target components
# ---------------------------------------------------------------------------


def test_relative_parts_preserves_intermediate_levels() -> None:
    """A registration two levels down keeps both levels, so the remediation
    target matches where the surrounding content actually migrates to.
    """
    outer = Path("C:/repos/jc/.var/devin-orchestrator/dispatches")
    inner = Path("C:/repos/jc/.var/devin-orchestrator/dispatches/reviews/pr-1384")
    assert _relative_parts(outer, inner) == ("reviews", "pr-1384")


def test_relative_parts_matches_case_and_separator_insensitively() -> None:
    """git emits forward slashes and may echo a different drive-letter case than
    the filesystem; neither may defeat the match.
    """
    outer = Path(r"C:\repos\JC\.var\devin-orchestrator\dispatches")
    inner = PurePosixPath("c:/repos/jc/.var/devin-orchestrator/dispatches/reviews/pr-1384")
    assert _relative_parts(outer, Path(str(inner))) == ("reviews", "pr-1384")


def test_relative_parts_returns_original_casing_not_folded() -> None:
    """The returned components go into a command an operator runs verbatim, so
    they must keep their real casing rather than the folded matching key.
    """
    outer = Path("C:/repos/jc/state")
    inner = Path("C:/repos/jc/state/Reviews/PR-1384")
    assert _relative_parts(outer, inner) == ("Reviews", "PR-1384")


def test_relative_parts_empty_for_same_path_and_for_non_descendant() -> None:
    same = Path("C:/repos/jc/state/dispatches")
    assert _relative_parts(same, same) == ()
    assert _relative_parts(same, Path("C:/repos/jc/state/dispatches-old/pr-1")) == ()
    assert _relative_parts(same, Path("C:/elsewhere/pr-1")) == ()
