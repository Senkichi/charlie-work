"""Dry-run previews versus the Apply pause gate (issue #3124).

``test_foreman_apply.py`` is attachment-point saturated, so this seams out:
pause stops effects, never previews. ``--dry-run --only`` is the probation
workflow of design §16 — a paused foreman's Apply walk must still evaluate
every action (so the operator reviews the plan it would apply when resumed)
while writing nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from charlie_work.foreman.apply import OUTCOME_APPLIED, apply_turn
from charlie_work.foreman.charter import Charter, CharterBounds
from charlie_work.foreman.items import load_items
from charlie_work.foreman.paths import applied_path
from charlie_work.foreman.state import ForemanState, read_state_snapshot, save_state
from charlie_work.local_issues import LocalFileGitHub

REPO = "acme/widgets"


def _setup(tmp_path: Path, *, state: str = "paused") -> tuple[Path, LocalFileGitHub, Charter]:
    froot = tmp_path / "foremen" / "pepe"
    froot.mkdir(parents=True)
    save_state(froot, ForemanState(state=state))
    repo_root = tmp_path / "repo"
    issues_dir = repo_root / ".issues"
    issues_dir.mkdir(parents=True)
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    charter = Charter(
        name="pepe", outcome="ship the outcome", repos=(REPO,), bounds=CharterBounds()
    )
    return froot, gh, charter


def test_dry_run_previews_a_paused_foreman(tmp_path: Path) -> None:
    """A paused foreman under ``--dry-run`` reports the plan it would apply —
    not all-skipped — and still writes nothing."""
    froot, gh, charter = _setup(tmp_path)
    turn = froot / "turns" / "t1"
    turn.mkdir(parents=True)
    (turn / "actions.json").write_text(
        json.dumps(
            {
                "turn_summary": "preview",
                "actions": [
                    {
                        "type": "file_issue",
                        "key": "a",
                        "repo": REPO,
                        "title": "Title a",
                        "body": "Body a.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = apply_turn(froot, "t1", charter=charter, gh=gh, dry_run=True)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_APPLIED
    assert "would file" in (outcome.detail or "")
    assert not result.value.stopped_by_pause
    # The preview leaves no trace: foreman still paused, no issue, no record.
    assert read_state_snapshot(froot).state == "paused"
    assert gh.issue_list() == []
    assert load_items(froot) == ()
    assert not applied_path(froot, "t1").exists()
