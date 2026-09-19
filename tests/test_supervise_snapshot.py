"""``take_snapshot``/``has_delta`` sidecar-and-verdict delta detection tests.

Split out of ``tests/test_supervise.py`` (issue #1562, Track 1) --
bodies are verbatim relocations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from charlie_work.supervise import has_delta, take_snapshot


def test_has_delta_no_change_returns_false(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    prs = tmp_path / "prs"
    prs.mkdir()
    snap1 = take_snapshot(sessions, prs)
    snap2 = take_snapshot(sessions, prs)
    assert has_delta(snap1, snap2) is False


def test_has_delta_live_count_change_returns_true(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    prs = tmp_path / "prs"
    prs.mkdir()
    snap1 = take_snapshot(sessions, prs)
    (sessions / "issue-1.json").write_text("{}", encoding="utf-8")
    snap2 = take_snapshot(sessions, prs)
    assert has_delta(snap1, snap2) is True


def test_has_delta_sidecar_mtime_change_returns_true(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    prs = tmp_path / "prs"
    prs.mkdir()
    sidecar = sessions / "issue-1.json"
    sidecar.write_text("{}", encoding="utf-8")
    snap1 = take_snapshot(sessions, prs)
    # Touch with a different mtime
    import os

    os.utime(sidecar, (sidecar.stat().st_atime + 1, sidecar.stat().st_mtime + 1))
    snap2 = take_snapshot(sessions, prs)
    assert has_delta(snap1, snap2) is True


def test_take_snapshot_verdict_mtimes_keys_on_pr_parent_not_filename(tmp_path: Path) -> None:
    """Regression for finding #1: verdict_mtimes must key on the PR-unique
    parent directory name ("pr-N"), not path.name (always the constant
    "review-decision.json"). Two PRs whose verdict files happen to share an
    identical mtime must still produce two distinct snapshot entries --
    keying on path.name alone would collapse them into a single set element
    (sets dedup identical tuples), silently erasing one PR's presence from
    the delta signal and letting a later rewrite go unnoticed.
    """
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    prs = tmp_path / "prs"
    prs.mkdir()
    pr1_dir = prs / "pr-1"
    pr1_dir.mkdir()
    pr2_dir = prs / "pr-2"
    pr2_dir.mkdir()
    verdict1 = pr1_dir / "review-decision.json"
    verdict2 = pr2_dir / "review-decision.json"
    verdict1.write_text('{"decision": "approved"}', encoding="utf-8")
    verdict2.write_text('{"decision": "approved"}', encoding="utf-8")

    import os

    shared_mtime = 1_700_000_000.0
    os.utime(verdict1, (shared_mtime, shared_mtime))
    os.utime(verdict2, (shared_mtime, shared_mtime))

    snap1 = take_snapshot(sessions, prs)
    assert len(snap1.verdict_mtimes) == 2, (
        "two PRs with identical verdict mtimes collapsed into one entry -- "
        "verdict_mtimes is keyed on path.name instead of the PR parent dir"
    )

    # Rewrite PR-1's verdict only; PR-2 is left untouched.
    os.utime(verdict1, (shared_mtime + 5, shared_mtime + 5))
    snap2 = take_snapshot(sessions, prs)
    assert has_delta(snap1, snap2) is True


def test_has_delta_new_verdict_file_returns_true(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    prs = tmp_path / "prs"
    prs.mkdir()
    snap1 = take_snapshot(sessions, prs)
    pr_dir = prs / "pr-1"
    pr_dir.mkdir()
    (pr_dir / "review-decision.json").write_text('{"decision": "approved"}', encoding="utf-8")
    snap2 = take_snapshot(sessions, prs)
    assert has_delta(snap1, snap2) is True


def test_take_snapshot_excludes_launch_failure_sidecar(tmp_path: Path) -> None:
    """Issue #266: a launch-failure sidecar (pid=None, error set) is not counted as live."""
    import json
    from charlie_work.devin_shell import SessionRecord

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    prs = tmp_path / "prs"
    prs.mkdir()

    sidecar = sessions / "issue-1.json"
    sidecar.write_text(
        json.dumps(
            SessionRecord(
                issue_number=1,
                branch="agent/issue-1-x",
                worktree_path="/tmp/worktree",
                prompt_path="/tmp/prompt.md",
                command=("devin",),
                pid=None,
                started_at="2024-01-01T00:00:00Z",
                log_path="/tmp/issue-1.log",
                error="worktree path already exists",
            ).to_dict()
        ),
        encoding="utf-8",
    )

    snap = take_snapshot(sessions, prs)
    assert snap.live_count == 0
    assert len(snap.sidecar_mtimes) == 1


def test_take_snapshot_counts_alive_workers_not_sidecar_files(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Issue #266: live_count reflects actual live workers, not raw file count."""
    import json
    from charlie_work.devin_shell import SessionRecord

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    prs = tmp_path / "prs"
    prs.mkdir()

    sidecar = sessions / "issue-1.json"
    sidecar.write_text(
        json.dumps(
            SessionRecord(
                issue_number=1,
                branch="agent/issue-1-x",
                worktree_path="/tmp/worktree",
                prompt_path="/tmp/prompt.md",
                command=("devin",),
                pid=12345,
                started_at="2024-01-01T00:00:00Z",
                log_path="/tmp/issue-1.log",
            ).to_dict()
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: True)
    snap = take_snapshot(sessions, prs)
    assert snap.live_count == 1
