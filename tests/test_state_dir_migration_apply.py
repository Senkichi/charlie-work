"""``apply_state_dir_migration`` tests -- the actuator.

Split out of ``tests/test_state_dir_migration.py`` (issue #1566, Track-1):
refusal on a not-ok or partially-blocked plan, the happy-path in-order
move, the default ``Path.rename`` mover, TOCTOU aborts when the
filesystem diverges from the plan snapshot, ``OSError``-as-value, and the
on-disk root-containment re-check.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

from _state_dir_migration_fixtures import DST_ROOT, SRC_ROOT
from charlie_work.state_migration import (
    MigrationChild,
    MigrationPlan,
    apply_state_dir_migration,
    plan_state_dir_migration,
)


# ---------------------------------------------------------------------------
# apply_state_dir_migration -- the actuator
# ---------------------------------------------------------------------------


def _recording_mover() -> tuple[list[tuple[Path, Path]], Callable[[Path, Path], None]]:
    """A ``mover`` fake that performs a real ``Path.rename`` and records each call.

    Performing the real rename (rather than a no-op) keeps on-disk state
    consistent across a multi-child run -- a later child's TOCTOU re-check
    must see the true post-move filesystem state, exactly as the real
    default mover would leave it.
    """
    calls: list[tuple[Path, Path]] = []

    def mover(src: Path, dst: Path) -> None:
        calls.append((src, dst))
        src.rename(dst)

    return calls, mover


def test_apply_refuses_when_plan_itself_is_not_ok() -> None:
    """A plan whose own ``.ok`` is False must never be actuated -- there is
    nothing safe to act on. This returns before any filesystem access, so
    the module-level (not-on-disk) path constants are fine here.
    """
    plan = MigrationPlan(
        src_root=SRC_ROOT,
        dst_root=DST_ROOT,
        children=(),
        ok=False,
        error="git worktree listing failed",
    )

    outcome = apply_state_dir_migration(plan)

    assert outcome.ok is False
    assert outcome.moved == ()
    assert outcome.aborted_at is None
    assert outcome.error is not None
    assert "git worktree listing failed" in outcome.error


def test_apply_refuses_when_blocked_children_present_names_both_in_error() -> None:
    """Refuses all-or-nothing when ANY child is blocked, naming every blocked
    child (not just the first) in ``.error``. A movable child sitting
    alongside blocked ones must not be partially moved.
    """
    blocked_a = MigrationChild(
        name="dispatches",
        src_path=SRC_ROOT / "dispatches",
        dst_path=DST_ROOT / "dispatches",
        disposition="blocked",
        reasons=("nested registered worktree",),
    )
    movable = MigrationChild(
        name="issues",
        src_path=SRC_ROOT / "issues",
        dst_path=DST_ROOT / "issues",
        disposition="move",
    )
    blocked_b = MigrationChild(
        name="events.db",
        src_path=SRC_ROOT / "events.db",
        dst_path=DST_ROOT / "events.db",
        disposition="blocked",
        reasons=("name collision",),
    )
    plan = MigrationPlan(
        src_root=SRC_ROOT, dst_root=DST_ROOT, children=(blocked_a, movable, blocked_b)
    )

    outcome = apply_state_dir_migration(plan)

    assert outcome.ok is False
    assert outcome.moved == ()
    assert outcome.aborted_at is None
    assert outcome.error is not None
    assert "dispatches" in outcome.error
    assert "events.db" in outcome.error


def test_apply_zero_movable_children_is_a_noop_success() -> None:
    """An empty plan (nothing left to migrate) is a legitimate steady state,
    not a failure -- mirrors ``gather_migration_inputs``'s own treatment of a
    missing src_root.
    """
    plan = MigrationPlan(src_root=SRC_ROOT, dst_root=DST_ROOT, children=())

    outcome = apply_state_dir_migration(plan)

    assert outcome.ok is True
    assert outcome.moved == ()
    assert outcome.error is None
    assert outcome.aborted_at is None


def test_apply_happy_path_moves_every_movable_child_in_order(tmp_path: Path) -> None:
    src_root = tmp_path / "src-state"
    dst_root = tmp_path / "dst-state"
    src_root.mkdir()
    names = ["issues", "prs", "logs"]
    for name in names:
        (src_root / name).mkdir()

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=[src_root / name for name in names],
        dst_names=[],
        registered_worktrees=[],
    )
    assert len(plan.movable) == 3

    calls, mover = _recording_mover()
    outcome = apply_state_dir_migration(plan, mover=mover)

    assert outcome.ok is True
    assert outcome.moved == tuple(names)
    assert outcome.error is None
    assert outcome.aborted_at is None
    assert [call[0].name for call in calls] == names
    for name in names:
        assert not (src_root / name).exists()
        assert (dst_root / name).exists()


def test_apply_default_mover_uses_path_rename_and_creates_dst_root(tmp_path: Path) -> None:
    """With no ``mover`` injected, the real default performs an actual
    ``Path.rename``, and a not-yet-existing ``dst_root`` is created for it --
    ``gather_migration_inputs`` documents that as this function's job, since
    a planning-only module cannot touch the filesystem.
    """
    src_root = tmp_path / "src-state"
    dst_root = tmp_path / "dst-state-not-created-yet"
    src_root.mkdir()
    (src_root / "issues").mkdir()
    (src_root / "issues" / "marker.txt").write_text("hello")

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=[src_root / "issues"],
        dst_names=[],
        registered_worktrees=[],
    )

    outcome = apply_state_dir_migration(plan)

    assert outcome.ok is True
    assert outcome.moved == ("issues",)
    assert dst_root.exists()
    assert (dst_root / "issues" / "marker.txt").read_text() == "hello"
    assert not (src_root / "issues").exists()


def test_apply_aborts_when_destination_appeared_since_planning(tmp_path: Path) -> None:
    """TOCTOU: the plan is a snapshot taken at planning time. If a child's
    destination has appeared on disk by actuation time, the whole run must
    abort rather than skip that one child and continue -- silently dropping
    "prs" while moving "issues" would be the same forbidden partial state as
    ignoring a blocked child outright.
    """
    src_root = tmp_path / "src-state"
    dst_root = tmp_path / "dst-state"
    src_root.mkdir()
    names = ["issues", "prs", "logs"]
    for name in names:
        (src_root / name).mkdir()

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=[src_root / name for name in names],
        dst_names=[],  # nothing collides at plan time
        registered_worktrees=[],
    )
    assert len(plan.movable) == 3

    # Simulate divergence between planning and actuation: "prs" (the second
    # child) now exists at its destination.
    dst_root.mkdir()
    (dst_root / "prs").mkdir()

    calls, mover = _recording_mover()
    outcome = apply_state_dir_migration(plan, mover=mover)

    assert outcome.ok is False
    assert outcome.moved == ("issues",)
    assert outcome.aborted_at == "prs"
    assert outcome.error is not None
    assert "prs" in outcome.error
    assert [call[0].name for call in calls] == ["issues"]
    # "logs" came after the aborted child and must be untouched.
    assert (src_root / "logs").exists()
    assert not (dst_root / "logs").exists()


def test_apply_aborts_when_source_vanished_since_planning(tmp_path: Path) -> None:
    """TOCTOU: if a child's source has vanished by actuation time (already
    moved or deleted by something else), the whole run must abort rather
    than silently skip it.
    """
    src_root = tmp_path / "src-state"
    dst_root = tmp_path / "dst-state"
    src_root.mkdir()
    names = ["issues", "prs", "logs"]
    for name in names:
        (src_root / name).mkdir()

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=[src_root / name for name in names],
        dst_names=[],
        registered_worktrees=[],
    )
    assert len(plan.movable) == 3

    # Simulate divergence: "prs" removed from src between planning and
    # actuation.
    (src_root / "prs").rmdir()

    calls, mover = _recording_mover()
    outcome = apply_state_dir_migration(plan, mover=mover)

    assert outcome.ok is False
    assert outcome.moved == ("issues",)
    assert outcome.aborted_at == "prs"
    assert outcome.error is not None
    assert "prs" in outcome.error
    assert [call[0].name for call in calls] == ["issues"]
    assert (src_root / "logs").exists()
    assert not (dst_root / "logs").exists()


def test_apply_oserror_from_mover_returns_ok_false_not_raised(tmp_path: Path) -> None:
    """Per CLAUDE.md ("errors from external processes come back as values,
    never raised"), an ``OSError`` from ``mover`` must come back as
    ``ok=False``, never propagate out of this function.
    """
    src_root = tmp_path / "src-state"
    dst_root = tmp_path / "dst-state"
    src_root.mkdir()
    names = ["issues", "prs"]
    for name in names:
        (src_root / name).mkdir()

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=[src_root / name for name in names],
        dst_names=[],
        registered_worktrees=[],
    )

    def boom_mover(src: Path, dst: Path) -> None:
        raise OSError("disk full")

    outcome = apply_state_dir_migration(plan, mover=boom_mover)

    assert outcome.ok is False
    assert outcome.moved == ()
    assert outcome.aborted_at == "issues"
    assert outcome.error is not None
    assert "disk full" in outcome.error
    # The fake mover raised before performing any real move.
    assert (src_root / "issues").exists()
    assert (src_root / "prs").exists()


def test_apply_refuses_when_a_child_escapes_the_roots_on_disk(tmp_path: Path) -> None:
    """A lexically-clean plan is still refused when the child resolves outside on disk.

    The planner is pure and can only judge shape; a symlink/junction escape exists
    only after resolution. Built by constructing the plan from real in-tree children
    and then pointing one child's ``src_path`` at an outside directory, which is what
    a resolved junction amounts to.
    """
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()
    (src_root / "issues").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=[src_root / "issues"],
        dst_names=[],
        registered_worktrees=[],
    )
    assert plan.blocked == ()

    escaped = dataclasses.replace(plan.children[0], src_path=outside)
    escaped_plan = dataclasses.replace(plan, children=(escaped,))

    moves: list[tuple[Path, Path]] = []
    outcome = apply_state_dir_migration(escaped_plan, mover=lambda s, d: moves.append((s, d)))

    assert outcome.ok is False
    assert outcome.aborted_at == "issues"
    assert outcome.error is not None
    assert "outside the migration roots" in outcome.error
    assert moves == []
