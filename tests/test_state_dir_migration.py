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

import subprocess
from pathlib import Path, PurePosixPath
from typing import Callable

import pytest

from charlie_work.state_migration import (
    MigrationChild,
    MigrationOutcome,
    MigrationPlan,
    _is_equal_or_nested,
    _normalize_path_key,
    _relative_parts,
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
