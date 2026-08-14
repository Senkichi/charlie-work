"""Tests for the pure state-DIR migration planner (state_migration.py).

Note on filename: this file is deliberately named ``test_state_dir_migration``
rather than the more obvious ``test_state_migration`` because that name is
already taken by a pre-existing, unrelated, committed test suite (PRs
#306/#321/#531) that covers ``charlie_work.state``'s ``load_state``/
``save_state`` fixture round-tripping against production ``state.json``
shapes -- a completely different meaning of "state migration" (schema
migration across orchestrator versions, not moving a state-dir tree between
roots). See this module's own docstring in ``src/charlie_work/state_migration.py``
for what this file actually tests.

Two regressions from the original hand-written migration probe are the
highest-value cases here (see that module docstring for the full incident
history):

1. A naive path comparison between ``git worktree list --porcelain`` output
   (always forward slashes) and ``Path.iterdir()`` output (backslashes on
   Windows) found zero matches, falsely reporting 74 registered worktrees as
   "orphaned". ``test_normalize_path_key_folds_separator_and_case`` and
   ``test_is_equal_or_nested_true_for_forward_slash_git_path_vs_backslash_disk_path``
   pin the fix directly.
2. A registered worktree can hide two levels deep inside a child not named
   "worktrees" at all (``dispatches/reviews/pr-1384``). The blocking rule
   must be derived from ``registered_worktrees`` at every depth, never a
   hardcoded list of "risky" names --
   ``test_plan_nested_registration_two_levels_deep_in_differently_named_child``
   pins this.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import pytest

from charlie_work.state_migration import (
    MigrationChild,
    MigrationOutcome,
    MigrationPlan,
    StateRewriteResult,
    _is_equal_or_nested,
    _normalize_path_key,
    _relative_parts,
    _rewrite_state_json_paths,
    _try_rewrite_path_string,
    _walk_and_rewrite,
    apply_state_dir_migration,
    gather_migration_inputs,
    plan_state_dir_migration,
)


REPO_ROOT = Path("C:/repos/job-cannon")
SRC_ROOT = Path("C:/repos/job-cannon/.var/devin-orchestrator")
DST_ROOT = Path("C:/repos/job-cannon/.var/charlie-work")


# --------------------------------------------------------------------------
# Value objects
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# _normalize_path_key / _is_equal_or_nested -- the separator/case defect
# --------------------------------------------------------------------------


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
    git_style = PurePosixPath("C:/Users/senki/repos/Job-Cannon/.claude/Worktrees/FOO")
    disk_style = Path(r"C:\Users\SENKI\repos\job-cannon\.claude\worktrees\foo")

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


# --------------------------------------------------------------------------
# plan_state_dir_migration -- rule 1: nested registration
# --------------------------------------------------------------------------


def test_plan_nested_registration_two_levels_deep_in_differently_named_child() -> None:
    """Reproduces the real job-cannon shape: two registered worktrees living
    two levels below a child named ``dispatches`` (not ``worktrees``).
    """
    dispatches = SRC_ROOT / "dispatches"
    pr_1384 = dispatches / "reviews" / "pr-1384"
    pr_1395 = dispatches / "reviews" / "pr-1395"

    plan = plan_state_dir_migration(
        repo_root=REPO_ROOT,
        src_root=SRC_ROOT,
        dst_root=DST_ROOT,
        src_children=[dispatches],
        dst_names=[],
        registered_worktrees=[pr_1384, pr_1395],
    )

    assert len(plan.children) == 1
    child = plan.children[0]
    assert child.blocked
    assert child.reasons == (
        f"2 registered git worktrees nested inside this child: {pr_1384}, {pr_1395}",
    )
    # The ``reviews/`` level is preserved: the registration must land where the rest of
    # that content migrates to (``<dst>/dispatches/reviews/pr-1384``), not be re-parented
    # by leaf name beside it (``<dst>/dispatches/pr-1384``).
    assert child.remediation == (
        f'git -C "{REPO_ROOT}" worktree move "{pr_1384}" '
        f'"{DST_ROOT / "dispatches" / "reviews" / "pr-1384"}"',
        f'git -C "{REPO_ROOT}" worktree move "{pr_1395}" '
        f'"{DST_ROOT / "dispatches" / "reviews" / "pr-1395"}"',
    )


def test_plan_nested_registration_is_child_itself_targets_child_dst_path() -> None:
    """When the registration IS the child (not nested below it), the
    remediation target must be the child's own destination path, not a
    doubled-up ``dst/child/child``.
    """
    worktrees = SRC_ROOT / "worktrees"

    plan = plan_state_dir_migration(
        repo_root=REPO_ROOT,
        src_root=SRC_ROOT,
        dst_root=DST_ROOT,
        src_children=[worktrees],
        dst_names=[],
        registered_worktrees=[worktrees],
    )

    child = plan.children[0]
    assert child.blocked
    assert child.remediation == (
        f'git -C "{REPO_ROOT}" worktree move "{worktrees}" "{DST_ROOT / "worktrees"}"',
    )


def test_plan_no_registration_nested_leaves_child_movable() -> None:
    issues = SRC_ROOT / "issues"
    unrelated = SRC_ROOT.parent / "some-other-repo-worktree"

    plan = plan_state_dir_migration(
        repo_root=REPO_ROOT,
        src_root=SRC_ROOT,
        dst_root=DST_ROOT,
        src_children=[issues],
        dst_names=[],
        registered_worktrees=[unrelated],
    )

    child = plan.children[0]
    assert child.movable
    assert child.reasons == ()
    assert child.remediation == ()


# --------------------------------------------------------------------------
# plan_state_dir_migration -- rule 2: name collision
# --------------------------------------------------------------------------


def test_plan_name_collision_blocks_with_case_insensitive_match() -> None:
    events_db = SRC_ROOT / "events.db"

    plan = plan_state_dir_migration(
        repo_root=REPO_ROOT,
        src_root=SRC_ROOT,
        dst_root=DST_ROOT,
        src_children=[events_db],
        dst_names=["EVENTS.DB"],
        registered_worktrees=[],
    )

    child = plan.children[0]
    assert child.blocked
    assert child.reasons == (
        f"name 'events.db' already exists in the destination at {DST_ROOT / 'events.db'}",
    )
    assert child.remediation == ()


def test_plan_name_collision_case_sensitive_miss_stays_movable() -> None:
    plan = plan_state_dir_migration(
        repo_root=REPO_ROOT,
        src_root=SRC_ROOT,
        dst_root=DST_ROOT,
        src_children=[SRC_ROOT / "issues"],
        dst_names=["prs"],
        registered_worktrees=[],
    )
    assert plan.children[0].movable


# --------------------------------------------------------------------------
# plan_state_dir_migration -- blocked for both reasons at once
# --------------------------------------------------------------------------


def test_plan_blocked_for_both_reasons_at_once() -> None:
    """A child can be blocked by rule 1 AND rule 2 simultaneously -- both
    reasons must be reported, never only the first one found.
    """
    dispatches = SRC_ROOT / "dispatches"
    nested = dispatches / "reviews" / "pr-1384"

    plan = plan_state_dir_migration(
        repo_root=REPO_ROOT,
        src_root=SRC_ROOT,
        dst_root=DST_ROOT,
        src_children=[dispatches],
        dst_names=["dispatches"],
        registered_worktrees=[nested],
    )

    child = plan.children[0]
    assert child.blocked
    assert len(child.reasons) == 2
    assert "registered git worktree" in child.reasons[0]
    assert "already exists in the destination" in child.reasons[1]
    assert len(child.remediation) == 1


# --------------------------------------------------------------------------
# plan_state_dir_migration -- rule 3: atomic sibling groups
# --------------------------------------------------------------------------


def test_plan_atomic_group_blocks_all_members_when_main_file_collides() -> None:
    """Exact shape from the coordinator's regression requirement: a WAL-mode
    SQLite database (``events.db`` + its ``-wal``/``-shm`` side files) must
    move as one unit or not at all. Blocking the main file must block both
    side files too, even though neither side file collides by name on its
    own.
    """
    names = ["events.db", "events.db-wal", "events.db-shm"]
    plan = plan_state_dir_migration(
        repo_root=REPO_ROOT,
        src_root=SRC_ROOT,
        dst_root=DST_ROOT,
        src_children=[SRC_ROOT / name for name in names],
        dst_names=["events.db"],
        registered_worktrees=[],
    )

    assert len(plan.blocked) == 3
    by_name = {child.name: child for child in plan.children}

    main = by_name["events.db"]
    assert main.blocked
    assert main.group == ("events.db", "events.db-shm", "events.db-wal")

    for side_name in ("events.db-wal", "events.db-shm"):
        side = by_name[side_name]
        assert side.blocked
        assert side.group == main.group
        assert any(
            reason.startswith("grouped with events.db, which is blocked:")
            for reason in side.reasons
        ), side.reasons


def test_plan_atomic_group_lone_side_file_without_main_is_not_grouped() -> None:
    """A ``-wal``/``-shm`` file with no matching main-file sibling present is
    not part of any group -- grouping only applies when the base name is
    actually among the children.
    """
    plan = plan_state_dir_migration(
        repo_root=REPO_ROOT,
        src_root=SRC_ROOT,
        dst_root=DST_ROOT,
        src_children=[SRC_ROOT / "orphan.db-wal"],
        dst_names=[],
        registered_worktrees=[],
    )
    child = plan.children[0]
    assert child.movable
    assert child.group == ()


def test_plan_atomic_group_all_movable_still_tagged_with_group() -> None:
    """When no member of a group is blocked, every member stays movable but
    still carries the ``group`` tag, so a CLI can render "these move
    together" without needing to re-derive the grouping itself.
    """
    names = ["events.db", "events.db-wal", "events.db-shm"]
    plan = plan_state_dir_migration(
        repo_root=REPO_ROOT,
        src_root=SRC_ROOT,
        dst_root=DST_ROOT,
        src_children=[SRC_ROOT / name for name in names],
        dst_names=[],
        registered_worktrees=[],
    )
    assert len(plan.movable) == 3
    for child in plan.children:
        assert child.group == ("events.db", "events.db-shm", "events.db-wal")


# --------------------------------------------------------------------------
# plan_state_dir_migration -- clean all-movable case
# --------------------------------------------------------------------------


def test_plan_clean_scenario_all_children_movable() -> None:
    names = ["issues", "prs", "logs", "sessions", "cross-family"]
    plan = plan_state_dir_migration(
        repo_root=REPO_ROOT,
        src_root=SRC_ROOT,
        dst_root=DST_ROOT,
        src_children=[SRC_ROOT / name for name in names],
        dst_names=[],
        registered_worktrees=[],
    )
    assert len(plan.movable) == len(names)
    assert plan.blocked == ()
    assert plan.ok is True
    assert plan.error is None


# --------------------------------------------------------------------------
# Ground-truth integration scenario (34 children, literal data, no filesystem)
# --------------------------------------------------------------------------


def test_plan_ground_truth_job_cannon_scenario_30_movable_4_blocked() -> None:
    """Reproduces the live measurement an independent oracle (built without
    importing charlie_work) produced for the real job-cannon host: 34 total
    children, 2 registered worktrees nested two levels inside ``dispatches``,
    and a name collision on both ``dispatches`` and ``events.db``.

    Before the atomic-group rule: 32 movable / 2 blocked. After it:
    30 movable / 4 blocked, because ``events.db-wal``/``events.db-shm`` join
    ``events.db``. ``dispatches`` is blocked for two independent reasons at
    once -- the case that proves the planner doesn't stop at the first
    reason found.

    The 30 "plain" children below are synthetic stand-ins for the real
    tree's uninteresting entries (their literal names are not significant to
    any assertion); the special four (``dispatches``, ``events.db`` + side
    files) and the two nested registrations are the exact production shape.
    """
    dispatches = SRC_ROOT / "dispatches"
    pr_1384 = dispatches / "reviews" / "pr-1384"
    pr_1395 = dispatches / "reviews" / "pr-1395"

    plain_names = [f"child-{i:02d}" for i in range(1, 31)]
    assert len(plain_names) == 30

    special_names = ["dispatches", "events.db", "events.db-wal", "events.db-shm"]
    all_names = special_names + plain_names
    assert len(all_names) == 34

    src_children = [SRC_ROOT / name for name in all_names]

    plan = plan_state_dir_migration(
        repo_root=REPO_ROOT,
        src_root=SRC_ROOT,
        dst_root=DST_ROOT,
        src_children=src_children,
        dst_names=["dispatches", "events.db"],
        registered_worktrees=[pr_1384, pr_1395],
    )

    assert len(plan.children) == 34
    assert len(plan.movable) == 30
    assert len(plan.blocked) == 4

    blocked_names = {child.name for child in plan.blocked}
    assert blocked_names == {"dispatches", "events.db", "events.db-wal", "events.db-shm"}

    by_name = {child.name: child for child in plan.children}
    dispatches_child = by_name["dispatches"]
    assert len(dispatches_child.reasons) == 2
    assert "2 registered git worktrees nested inside this child" in dispatches_child.reasons[0]
    assert "already exists in the destination" in dispatches_child.reasons[1]

    expected_move_1384 = (
        f'git -C "{REPO_ROOT}" worktree move "{pr_1384}" '
        f'"{DST_ROOT / "dispatches" / "reviews" / "pr-1384"}"'
    )
    expected_move_1395 = (
        f'git -C "{REPO_ROOT}" worktree move "{pr_1395}" '
        f'"{DST_ROOT / "dispatches" / "reviews" / "pr-1395"}"'
    )
    assert expected_move_1384 in dispatches_child.remediation
    assert expected_move_1395 in dispatches_child.remediation

    for side_name in ("events.db-wal", "events.db-shm"):
        assert any(
            reason.startswith("grouped with events.db, which is blocked:")
            for reason in by_name[side_name].reasons
        )

    # Every plain child is movable and carries no group tag.
    for name in plain_names:
        plain_child = by_name[name]
        assert plain_child.movable
        assert plain_child.group == ()


# --------------------------------------------------------------------------
# gather_migration_inputs -- the impure gatherer
# --------------------------------------------------------------------------


def test_gather_migration_inputs_missing_src_root_is_ok_with_zero_children(
    tmp_path: Path,
) -> None:
    """A missing src_root is a normal steady state (nothing left to migrate),
    not an error -- ``ok`` stays ``True`` with zero children.
    """
    src_root = tmp_path / "does-not-exist"
    dst_root = tmp_path / "dst"

    plan = gather_migration_inputs(repo_root=tmp_path, src_root=src_root, dst_root=dst_root)

    assert plan.ok is True
    assert plan.error is None
    assert plan.children == ()
    assert plan.src_root == src_root
    assert plan.dst_root == dst_root


def test_gather_migration_inputs_git_listing_failure_returns_error_not_exception(
    tmp_path: Path,
) -> None:
    """If the registered-worktree listing fails, planning must not proceed as
    though zero worktrees were registered -- that is exactly the unsafe
    assumption this module exists to prevent. The error comes back as a
    value on ``.error``, never an exception (CLAUDE.md: errors as values).
    """
    repo_root = tmp_path / "not-a-git-repo"
    repo_root.mkdir()
    src_root = tmp_path / "src-state"
    src_root.mkdir()
    (src_root / "issues").mkdir()
    dst_root = tmp_path / "dst-state"

    plan = gather_migration_inputs(repo_root=repo_root, src_root=src_root, dst_root=dst_root)

    assert plan.ok is False
    assert plan.children == ()
    assert plan.error is not None
    assert "could not list registered git worktrees" in plan.error


def test_gather_migration_inputs_missing_dst_root_treated_as_no_existing_entries(
    tmp_path: Path,
) -> None:
    """A dst_root that does not exist yet has no entries to collide with --
    this is the actuator's job to create, not an error here. This still
    exercises the git-listing path, so it uses a real (empty) git repo.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo_root, check=True)

    src_root = tmp_path / "src-state"
    src_root.mkdir()
    (src_root / "issues").mkdir()
    dst_root = tmp_path / "dst-state-not-created-yet"

    plan = gather_migration_inputs(repo_root=repo_root, src_root=src_root, dst_root=dst_root)

    assert plan.ok is True
    assert plan.error is None
    assert len(plan.children) == 1
    assert plan.children[0].movable


# --------------------------------------------------------------------------
# Relative-path preservation (remediation targets)
# --------------------------------------------------------------------------


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


def test_plan_nested_registration_deeper_than_one_level_keeps_full_relative_path() -> None:
    """End-to-end via the planner: the emitted ``git worktree move`` target must
    contain every intermediate level, not just the leaf name.
    """
    child = SRC_ROOT / "dispatches"
    reg = child / "reviews" / "pr-1384"
    plan = plan_state_dir_migration(
        repo_root=REPO_ROOT,
        src_root=SRC_ROOT,
        dst_root=DST_ROOT,
        src_children=[child],
        dst_names=[],
        registered_worktrees=[reg],
    )
    (blocked,) = plan.blocked
    expected_target = DST_ROOT / "dispatches" / "reviews" / "pr-1384"
    assert blocked.remediation == (
        f'git -C "{REPO_ROOT}" worktree move "{reg}" "{expected_target}"',
    )


# --------------------------------------------------------------------------
# Fail-closed on an unusable worktree listing
# --------------------------------------------------------------------------


def test_gather_migration_inputs_fails_closed_on_entry_without_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry whose "worktree" value is not a Path must abort planning, not be
    filtered out.

    Silently dropping it degrades to "zero worktrees registered", which is the
    single reading that lets a directory move break a live registration -- the
    exact failure this module exists to prevent. Unknown must never read as safe.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    src_root = tmp_path / "src-state"
    src_root.mkdir()
    (src_root / "dispatches").mkdir()
    dst_root = tmp_path / "dst-state"

    monkeypatch.setattr(
        "charlie_work.state_migration._list_worktrees_porcelain",
        lambda repo_root: ([{"worktree": "C:/not/a/Path/object"}], None),
    )

    plan = gather_migration_inputs(repo_root=repo_root, src_root=src_root, dst_root=dst_root)

    assert plan.ok is False
    assert plan.children == ()
    assert plan.error is not None
    assert "no usable path" in plan.error


# --------------------------------------------------------------------------
# apply_state_dir_migration -- the actuator
# --------------------------------------------------------------------------


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


def test_migration_outcome_is_frozen() -> None:
    """MigrationOutcome follows the project's frozen-dataclass rule."""
    outcome = MigrationOutcome(ok=True, moved=("issues",))
    with pytest.raises(Exception):
        outcome.ok = False  # type: ignore[misc]


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


def test_plan_blocks_child_whose_source_is_outside_src_root(tmp_path: Path) -> None:
    """A caller-supplied path from outside the tree is blocked, not moved.

    ``plan_state_dir_migration`` is public and takes ``src_children`` verbatim, and
    ``MigrationChild.src_path`` is that path unchanged. Without this rule the actuator
    would happily relocate any path handed to it.
    """
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()
    outsider = tmp_path / "elsewhere" / "secrets"
    outsider.parent.mkdir()
    outsider.mkdir()

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=[outsider],
        dst_names=[],
        registered_worktrees=[],
    )

    assert [child.name for child in plan.blocked] == ["secrets"]
    assert any("not inside the source root" in reason for reason in plan.blocked[0].reasons)


def test_plan_blocks_dotdot_child_that_climbs_out_of_dst_root(tmp_path: Path) -> None:
    """``Path("..").name`` is ``".."``, so ``dst_root / name`` escapes the destination."""
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=[src_root / ".."],
        dst_names=[],
        registered_worktrees=[],
    )

    assert len(plan.blocked) == 1
    assert any("escapes the destination root" in r for r in plan.blocked[0].reasons)


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


# --------------------------------------------------------------------------
# Issue #735: embedded-path rewrite inside state.json
# --------------------------------------------------------------------------
#
# ``apply_state_dir_migration`` moves the tree but must also rewrite every
# embedded absolute path inside ``state.json`` so the file is internally
# consistent with its new location. The tests below cover the structural
# walk, the existence verification, the count reporting, and the refusal
# when a hit's rewritten target does not exist.


def test_state_rewrite_result_is_frozen() -> None:
    """StateRewriteResult follows the project's frozen-dataclass rule."""
    result = StateRewriteResult(ok=True, rewritten=3)
    with pytest.raises(Exception):
        result.ok = False  # type: ignore[misc]


def test_migration_outcome_has_rewritten_paths_field_default_zero() -> None:
    """The new field exists and defaults to 0 when not specified."""
    outcome = MigrationOutcome(ok=True, moved=("issues",))
    assert outcome.rewritten_paths == 0


def test_try_rewrite_path_string_non_path_string_is_not_a_hit() -> None:
    """A string that is not under src_root is returned unchanged, count 0."""
    src = Path("C:/repos/x/.var/old")
    dst = Path("C:/repos/x/.var/new")
    result, count, error = _try_rewrite_path_string("not-a-path", src, dst)
    assert result == "not-a-path"
    assert count == 0
    assert error is None


def test_try_rewrite_path_string_issue_title_containing_prefix_is_not_a_hit(
    tmp_path: Path,
) -> None:
    """An issue title that merely *contains* the old-root string but does not
    start with it is not a hit -- the exact-prefix check rejects it, exactly
    the hazard a ``str.replace`` would get wrong.
    """
    src = tmp_path / "old-state"
    dst = tmp_path / "new-state"
    src.mkdir()
    dst.mkdir()
    title = f"Fix bug in {src} module"
    result, count, error = _try_rewrite_path_string(title, src, dst)
    assert result == title
    assert count == 0
    assert error is None


def test_try_rewrite_path_string_sibling_prefix_is_not_a_hit(tmp_path: Path) -> None:
    """A path like ``<src>-backup/...`` is not under ``src`` -- the exact-prefix
    check (separator after the root) rejects it, while a bare ``startswith``
    would wrongly match.
    """
    src = tmp_path / "old-state"
    dst = tmp_path / "new-state"
    src.mkdir()
    dst.mkdir()
    sibling = src.parent / "old-state-backup"
    sibling.mkdir()
    candidate = str(sibling / "file.txt")
    result, count, error = _try_rewrite_path_string(candidate, src, dst)
    assert result == candidate
    assert count == 0
    assert error is None


def test_walk_and_rewrite_preserves_non_string_values() -> None:
    """Ints, floats, bools, None, and dict keys are passed through untouched."""
    src = Path("C:/repos/x/.var/old")
    dst = Path("C:/repos/x/.var/new")
    data: dict[str, Any] = {
        "count": 42,
        "ratio": 3.14,
        "flag": True,
        "nothing": None,
        "title": "some issue title",
    }
    new_data, count, error = _walk_and_rewrite(data, src, dst)
    assert error is None
    assert count == 0
    assert new_data == data


def test_walk_and_rewrite_does_not_touch_dict_keys(tmp_path: Path) -> None:
    """Dict keys are never rewritten -- only string *values* are candidates."""
    src = tmp_path / "old-state"
    dst = tmp_path / "new-state"
    src.mkdir()
    dst.mkdir()
    # The key "old-state" is not a path under src_root; the value is.
    target_file = src / "dispatches" / "prompt.md"
    target_file.parent.mkdir(parents=True)
    target_file.write_text("x")
    dst_target = dst / "dispatches" / "prompt.md"
    dst_target.parent.mkdir(parents=True)
    dst_target.write_text("x")
    data = {"old-state": str(target_file)}
    new_data, count, error = _walk_and_rewrite(data, src, dst)
    assert error is None
    assert count == 1
    assert "old-state" in new_data  # key unchanged
    assert new_data["old-state"] == str(dst_target)


def test_walk_and_rewrite_rewrites_path_strings_inside_non_empty_list(tmp_path: Path) -> None:
    """The list-handling branch of ``_walk_and_rewrite`` must descend into a
    non-empty list and rewrite path-string elements under ``src_root``.

    Real event payloads store paths inside the ``events`` array -- a list of
    event dicts whose ``data`` values include path strings like
    ``reviews_dir`` (see ``workflow._mark_review_checkout_removal_failed``).
    A walk that only recursed into dicts would miss every path embedded in
    that list; this test pins the list branch with the exact shape: a list of
    event dicts, each carrying a path-string value under the old root.
    """
    src = tmp_path / "old-state"
    dst = tmp_path / "new-state"
    src.mkdir()
    dst.mkdir()
    # Two event dicts, each with a ``reviews_dir`` path under src_root -- the
    # real ``review_checkout_removal_failed`` event payload shape.
    reviews_dir_1 = src / "dispatches" / "reviews" / "pr-1"
    reviews_dir_2 = src / "dispatches" / "reviews" / "pr-2"
    reviews_dir_1.mkdir(parents=True)
    reviews_dir_2.mkdir(parents=True)
    dst_reviews_dir_1 = dst / "dispatches" / "reviews" / "pr-1"
    dst_reviews_dir_2 = dst / "dispatches" / "reviews" / "pr-2"
    dst_reviews_dir_1.mkdir(parents=True)
    dst_reviews_dir_2.mkdir(parents=True)
    data = {
        "version": 1,
        "events": [
            {
                "kind": "review_checkout_removal_failed",
                "data": {"reviews_dir": str(reviews_dir_1)},
            },
            {
                "kind": "review_checkout_removal_failed",
                "data": {"reviews_dir": str(reviews_dir_2)},
            },
        ],
    }
    new_data, count, error = _walk_and_rewrite(data, src, dst)
    assert error is None
    assert count == 2
    events = new_data["events"]
    assert len(events) == 2
    assert events[0]["data"]["reviews_dir"] == str(dst_reviews_dir_1)
    assert events[1]["data"]["reviews_dir"] == str(dst_reviews_dir_2)
    # Non-path list elements (e.g. an event with no path) are passed through.
    assert events[0]["kind"] == "review_checkout_removal_failed"


def test_walk_and_rewrite_list_branch_propagates_rewrite_failure(tmp_path: Path) -> None:
    """A path string inside a list whose rewritten target does not exist must
    fail the whole walk -- the caller must not save a partially-rewritten tree.

    Exercises the list branch's error-return path (the ``if error is not
    None: return value, 0, error`` inside the list loop), which a walk that
    only tested dict-level failures would never reach.
    """
    src = tmp_path / "old-state"
    dst = tmp_path / "new-state"
    src.mkdir()
    dst.mkdir()
    # A path under src_root whose target under dst_root does NOT exist -- the
    # existence check in ``_try_rewrite_path_string`` must refuse it.
    bogus = src / "dispatches" / "missing.md"
    bogus.parent.mkdir(parents=True)
    data = {
        "version": 1,
        "events": [
            {"kind": "review_checkout_removal_failed", "data": {"reviews_dir": str(bogus)}},
        ],
    }
    new_data, count, error = _walk_and_rewrite(data, src, dst)
    assert error is not None
    assert "does not exist" in error
    assert count == 0
    # On error, the original value is returned unchanged -- no partial rewrite.
    assert new_data == data


def test_rewrite_state_json_paths_missing_file_is_ok_zero(tmp_path: Path) -> None:
    """A missing state.json is not an error -- nothing to rewrite."""
    state_path = tmp_path / "nonexistent-state.json"
    result = _rewrite_state_json_paths(state_path, tmp_path / "old", tmp_path / "new")
    assert result.ok is True
    assert result.rewritten == 0
    assert result.error is None


def _make_state_with_embedded_paths(src_root: Path, *, pr_count: int = 1) -> dict[str, Any]:
    """Build a state.json-shaped dict with embedded absolute paths under src_root.

    Mirrors the real field names from the job-cannon incident: ``prompt_path``,
    ``decision_path``, ``cross_family_report``, and ``verdict_source``. Each
    path points at a file that exists under ``src_root`` so the post-move
    existence check passes.
    """
    prs: dict[str, Any] = {}
    for i in range(1, pr_count + 1):
        pr_dir = src_root / "dispatches" / "reviews" / f"pr-{i}"
        pr_dir.mkdir(parents=True, exist_ok=True)
        prompt = pr_dir / "prompt.md"
        prompt.write_text("prompt")
        decision = pr_dir / "decision.md"
        decision.write_text("decision")
        cross_family = pr_dir / "cross-family-report.md"
        cross_family.write_text("report")
        prs[str(i)] = {
            "number": i,
            "title": f"PR {i} -- has {src_root} in title for non-hit test",
            "prompt_path": str(prompt),
            "decision_path": str(decision),
            "cross_family_report": str(cross_family),
            "verdict_source": str(decision),
        }
    return {
        "version": 1,
        "issues": {"101": {"number": 101, "title": "some issue"}},
        "prs": prs,
        "events": [],
    }


def test_apply_rewrites_embedded_paths_in_state_json(tmp_path: Path) -> None:
    """The core #735 regression: after moving children, every embedded absolute
    path inside state.json is rewritten from the old root to the new one, the
    count is reported in ``rewritten_paths``, and the file on disk reflects
    the rewrite.
    """
    src_root = tmp_path / "src-state"
    dst_root = tmp_path / "dst-state"
    src_root.mkdir()

    state_data = _make_state_with_embedded_paths(src_root, pr_count=2)
    state_file = src_root / "state.json"
    state_file.write_text(json.dumps(state_data), encoding="utf-8")

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=list(src_root.iterdir()),
        dst_names=[],
        registered_worktrees=[],
    )
    assert plan.blocked == ()

    outcome = apply_state_dir_migration(plan)

    assert outcome.ok is True
    # 2 PRs × 4 path fields each = 8 rewrites
    assert outcome.rewritten_paths == 8
    assert outcome.error is None

    # Verify the file on disk: every path field now names dst_root.
    rewritten = json.loads((dst_root / "state.json").read_text(encoding="utf-8"))
    for pr_data in rewritten["prs"].values():
        for field in ("prompt_path", "decision_path", "cross_family_report", "verdict_source"):
            value = pr_data[field]
            assert str(dst_root) in value, f"{field} still names old root: {value}"
            assert str(src_root) not in value, f"{field} still names old root: {value}"
            assert Path(value).exists(), f"{field} rewritten path does not exist: {value}"

    # Non-path strings (issue/PR titles) are untouched even if they contain
    # the old-root string -- the exact-prefix check rejects them.
    for pr_data in rewritten["prs"].values():
        assert str(src_root) in pr_data["title"], "title should be unchanged"

    # Key sets are byte-identical before and after (the walk must not add or
    # drop keys -- mirrors the job-cannon remediation verification).
    assert set(rewritten["prs"]) == set(state_data["prs"])
    assert set(rewritten["issues"]) == set(state_data["issues"])


def test_apply_no_state_json_reports_zero_rewrites(tmp_path: Path) -> None:
    """When there is no state.json among the moved children, rewritten_paths
    is 0 and the migration still succeeds.
    """
    src_root = tmp_path / "src-state"
    dst_root = tmp_path / "dst-state"
    src_root.mkdir()
    (src_root / "issues").mkdir()
    (src_root / "logs").mkdir()

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=[src_root / "issues", src_root / "logs"],
        dst_names=[],
        registered_worktrees=[],
    )

    outcome = apply_state_dir_migration(plan)

    assert outcome.ok is True
    assert outcome.rewritten_paths == 0


def test_apply_state_json_with_no_embedded_paths_reports_zero(tmp_path: Path) -> None:
    """A state.json with no paths under the old root yields 0 rewrites but
    is still a successful migration.
    """
    src_root = tmp_path / "src-state"
    dst_root = tmp_path / "dst-state"
    src_root.mkdir()
    (src_root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=[src_root / "state.json"],
        dst_names=[],
        registered_worktrees=[],
    )

    outcome = apply_state_dir_migration(plan)

    assert outcome.ok is True
    assert outcome.rewritten_paths == 0


def test_apply_rewrite_failure_returns_ok_false_with_moved(tmp_path: Path) -> None:
    """If the state rewrite fails (a hit whose rewritten target does not
    exist), the outcome is ``ok=False`` with ``moved`` listing what was
    moved -- the children are already on the new root, so this is an
    incomplete migration needing manual attention, not a rollback.
    """
    src_root = tmp_path / "src-state"
    dst_root = tmp_path / "dst-state"
    src_root.mkdir()

    # Create a state.json with a path under src_root, but do NOT create the
    # file it points at. After the move, the rewritten path will not exist.
    bogus_path = src_root / "dispatches" / "missing.md"
    state_data = {
        "version": 1,
        "issues": {},
        "prs": {"1": {"prompt_path": str(bogus_path)}},
        "events": [],
    }
    (src_root / "state.json").write_text(json.dumps(state_data), encoding="utf-8")

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=[src_root / "state.json"],
        dst_names=[],
        registered_worktrees=[],
    )

    outcome = apply_state_dir_migration(plan)

    assert outcome.ok is False
    assert outcome.moved == ("state.json",)
    assert outcome.rewritten_paths == 0
    assert outcome.error is not None
    assert "path rewrite failed" in outcome.error
    assert "does not exist" in outcome.error


def test_apply_uses_injected_state_rewriter_seam(tmp_path: Path) -> None:
    """The ``state_rewriter`` seam is injectable for testability, mirroring the
    ``mover`` seam. A fake that reports 5 rewrites is reflected in the outcome.
    """
    src_root = tmp_path / "src-state"
    dst_root = tmp_path / "dst-state"
    src_root.mkdir()
    (src_root / "issues").mkdir()

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=[src_root / "issues"],
        dst_names=[],
        registered_worktrees=[],
    )

    def fake_rewriter(state_path: Path, src: Path, dst: Path) -> StateRewriteResult:
        return StateRewriteResult(ok=True, rewritten=5)

    outcome = apply_state_dir_migration(plan, state_rewriter=fake_rewriter)

    assert outcome.ok is True
    assert outcome.rewritten_paths == 5


def test_apply_injected_state_rewriter_failure_propagates(tmp_path: Path) -> None:
    """A failing injected state_rewriter makes the whole outcome ``ok=False``."""
    src_root = tmp_path / "src-state"
    dst_root = tmp_path / "dst-state"
    src_root.mkdir()
    (src_root / "issues").mkdir()

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=[src_root / "issues"],
        dst_names=[],
        registered_worktrees=[],
    )

    def failing_rewriter(state_path: Path, src: Path, dst: Path) -> StateRewriteResult:
        return StateRewriteResult(ok=False, error="lock timeout")

    outcome = apply_state_dir_migration(plan, state_rewriter=failing_rewriter)

    assert outcome.ok is False
    assert outcome.moved == ("issues",)
    assert outcome.rewritten_paths == 0
    assert "lock timeout" in (outcome.error or "")
