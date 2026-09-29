"""A devin launch retires the previous claude session's events.jsonl (issue #2036).

The claude-code adapter writes ``issue-N[-review|-rework].events.jsonl`` into
the same sessions dir a devin session uses, and the verdict / miss-cause /
session-summary / metrics readers in ``misc_review_verdicts`` read that path
regardless of harness. Before #2036 only the claude launcher rotated it, so a
devin reviewer that died on quota exhaustion was classified from the previous
claude round's transcript (``turn_limit_summary_posted``) and a stale summary
was posted to the PR.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work import devin_shell
from charlie_work.claude_code import _events_path
from charlie_work.devin_shell import launch_devin_session

_STALE = '{"type":"result","subtype":"success","result":"round-1 claude verdict"}\n'


@pytest.mark.parametrize(
    ("review", "rework"),
    [(True, False), (False, True), (False, False)],
    ids=["review", "rework", "dispatch"],
)
def test_devin_launch_rotates_previous_claude_events_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, review: bool, rework: bool
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    stale = _events_path(sessions_dir, 12, rework=rework, review=review)
    stale.write_text(_STALE, encoding="utf-8")

    def failing_create_worktree(*args: object, **kwargs: object) -> None:
        raise OSError("no worktree in this test")

    monkeypatch.setattr(devin_shell, "create_worktree", failing_create_worktree)

    # review=True without head_sha fails before any checkout; the worker lanes
    # fail at create_worktree. Either way the launch returns an error record,
    # and the rotation must already have happened.
    record = launch_devin_session(
        12,
        "agent/issue-12",
        tmp_path / "prompt.md",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        rework=rework,
        review=review,
    )

    assert record.error
    assert not stale.exists(), "stale claude events file is still readable as this session's"
    rotated = stale.with_suffix(stale.suffix + ".1")
    assert rotated.read_text(encoding="utf-8") == _STALE


def test_devin_launch_leaves_other_lanes_events_files_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: a review launch retires only the review lane's file."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    dispatch_events = _events_path(sessions_dir, 12)
    dispatch_events.write_text(_STALE, encoding="utf-8")

    launch_devin_session(
        12,
        "agent/issue-12",
        tmp_path / "prompt.md",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
    )

    assert dispatch_events.read_text(encoding="utf-8") == _STALE
