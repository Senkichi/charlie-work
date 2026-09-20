"""``plan_state_dir_migration`` / ``gather_migration_inputs`` tests.

Split out of ``tests/test_state_dir_migration.py`` (issue #1566, Track-1):
the pure planner's blocking rules (nested worktree registration, name
collision, atomic sibling groups, root containment), the clean all-movable
case, the 34-child ground-truth integration scenario, and the impure
gatherer that feeds it (fail-closed on an unusable worktree listing).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from _state_dir_migration_fixtures import DST_ROOT, REPO_ROOT, SRC_ROOT
from charlie_work.state_migration import (
    gather_migration_inputs,
    plan_state_dir_migration,
)


# ---------------------------------------------------------------------------
# plan_state_dir_migration -- rule 1: nested registration
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# plan_state_dir_migration -- rule 2: name collision
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# plan_state_dir_migration -- blocked for both reasons at once
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# plan_state_dir_migration -- rule 3: atomic sibling groups
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# plan_state_dir_migration -- clean all-movable case
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Ground-truth integration scenario (34 children, literal data, no filesystem)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# plan_state_dir_migration -- root containment
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# gather_migration_inputs -- the impure gatherer
# ---------------------------------------------------------------------------


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
