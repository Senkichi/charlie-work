"""Table-driven ``classify`` -- one case per row of the spec's classification table."""

from __future__ import annotations

from dataclasses import replace

import pytest

from charlie_work.main_sync_classify import (
    CheckoutState,
    IncomingCommit,
    SyncDecision,
    classify,
    instruction_paths,
    split_at_first_touch,
)

BEHIND = CheckoutState(
    default_branch="main",
    current_branch="main",
    local_sha="a" * 40,
    origin_sha="b" * 40,
    merge_base="a" * 40,
    ahead=0,
    behind=2,
    incoming_paths=("src/x.py", "uv.lock"),
)


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        pytest.param(
            replace(BEHIND, origin_sha="a" * 40, behind=0), SyncDecision("none"), id="up-to-date"
        ),
        pytest.param(BEHIND, SyncDecision("ff"), id="behind-clean"),
        pytest.param(
            replace(BEHIND, dirty_tracked=("README.md",), untracked=("notes.txt",)),
            SyncDecision("ff"),
            id="behind-dirty-elsewhere",
        ),
        pytest.param(
            replace(BEHIND, current_branch="feature"),
            SyncDecision("ref_update"),
            id="other-branch",
        ),
        pytest.param(
            replace(BEHIND, current_branch="feature", default_checked_out_elsewhere=True),
            SyncDecision("skip", "checked_out_elsewhere"),
            id="other-branch-main-in-linked-worktree",
        ),
        pytest.param(
            replace(BEHIND, merge_base=None, ahead=5),
            SyncDecision("skip", "unrelated_history"),
            id="unrelated",
        ),
        pytest.param(replace(BEHIND, ahead=1), SyncDecision("skip", "diverged"), id="diverged"),
        pytest.param(
            replace(BEHIND, ahead=1, behind=0, origin_sha="c" * 40),
            SyncDecision("skip", "diverged"),
            id="ahead-only",
        ),
        pytest.param(
            replace(BEHIND, dirty_tracked=("uv.lock", "README.md")),
            SyncDecision("skip", "blocked", ("uv.lock",)),
            id="dirty-overlap",
        ),
        pytest.param(
            replace(BEHIND, untracked=("src/x.py",)),
            SyncDecision("skip", "blocked", ("src/x.py",)),
            id="untracked-overlap",
        ),
        pytest.param(
            replace(BEHIND, operation_in_progress="rebase_in_progress"),
            SyncDecision("skip", "rebase_in_progress"),
            id="rebase",
        ),
        pytest.param(
            replace(BEHIND, index_locked=True), SyncDecision("skip", "index_locked"), id="lock"
        ),
        pytest.param(
            replace(BEHIND, current_branch=None),
            SyncDecision("skip", "detached_head"),
            id="detached",
        ),
    ],
)
def test_classify(state: CheckoutState, expected: SyncDecision) -> None:
    assert classify(state) == expected


def test_up_to_date_wins_over_every_skip() -> None:
    state = replace(BEHIND, origin_sha="a" * 40, behind=0, index_locked=True, current_branch=None)
    assert classify(state) == SyncDecision("none")


COMMITS = (
    IncomingCommit("c1", ("README.md",)),
    IncomingCommit("c2", ("src/a.py",)),
    IncomingCommit("c3", ("uv.lock", "src/b.py")),
    IncomingCommit("c4", ("README.md",)),
)


def test_split_stops_before_first_touching_commit() -> None:
    assert split_at_first_touch(COMMITS, ("uv.lock",)) == ("c2", True)


def test_split_first_commit_touching_leaves_nothing_safe() -> None:
    assert split_at_first_touch(COMMITS, ("README.md",)) == (None, True)


def test_split_no_touch_means_whole_range_without_post_sync() -> None:
    assert split_at_first_touch(COMMITS, ("pyproject.toml",)) == (None, False)


def test_split_empty_patterns_means_every_commit_touches() -> None:
    assert split_at_first_touch(COMMITS, ()) == (None, True)


def test_split_glob_crosses_directories() -> None:
    assert split_at_first_touch(COMMITS, ("src/*",)) == ("c1", True)


def test_split_empty_range() -> None:
    assert split_at_first_touch((), ("uv.lock",)) == (None, False)


def test_instruction_paths() -> None:
    paths = [
        "CLAUDE.md",
        "pkg/AGENTS.md",
        ".claude/settings.json",
        "sub/.claude/skills/x.md",
        "src/claude.py",
        "docs/CLAUDE.md.bak",
        "CLAUDE.md",
    ]
    assert instruction_paths(paths) == (
        ".claude/settings.json",
        "CLAUDE.md",
        "pkg/AGENTS.md",
        "sub/.claude/skills/x.md",
    )
