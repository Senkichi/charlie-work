"""Shared dead-session / orphaned-worker test fixtures.

Helpers hoisted out of ``tests/test_charlie_work.py`` (issue #1551,
Track-1 wave 5/8) because they are used by the dead-session,
orphaned-worker, and stall/reap sibling test modules: the flat
review-decision writer, the ``git`` subprocess wrapper, the dead
session sidecar writer, and the classify-state scaffold.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from charlie_work.devin_shell import SessionRecord


def _write_flat_review_decision(
    paths: Any, pr_number: int, decision: str, reviewed_head_sha: str | None
) -> None:
    """Write a flat ``prs/pr-N/review-decision.json`` matching a test's
    ``state["prs"][...]`` fixture.

    Issue #1362 Stage 1: control-flow reads of a PR's review decision go
    through the file-first ``review_decision`` reader now, not
    ``state.json``'s ``decision``/``reviewed_head_sha`` fields. A fixture
    that only writes those fields into ``state.json`` no longer drives the
    behavior it used to -- the flat file must exist and agree, or the
    reader reports ``missing``.
    """
    pr_dir = paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"decision": decision}
    if reviewed_head_sha is not None:
        payload["reviewed_head_sha"] = reviewed_head_sha
    (pr_dir / "review-decision.json").write_text(json.dumps(payload), encoding="utf-8")


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _write_dead_session_sidecar(
    sessions_dir: Path, issue_number: int, branch: str, worktree_path: Path
) -> None:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    record = SessionRecord(
        issue_number=issue_number,
        branch=branch,
        worktree_path=str(worktree_path),
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(sessions_dir / f"issue-{issue_number}.log"),
        error=None,
    )
    sidecar_path = sessions_dir / f"issue-{issue_number}.json"
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")


def _make_classify_state(tmp_path: Path) -> tuple[Path, Path]:
    """Create a state file and sessions dir under tmp_path, return (sessions_dir, state_file)."""
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": []}), encoding="utf-8")
    return sessions_dir, state_file
