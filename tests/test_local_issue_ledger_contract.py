"""ci-fleet's ledger files ``local/*`` issues that this tracker then edits.

The ledger (``ci_fleet.ledger.issues``) writes issue files for nightly breaks,
regressions and consolidation tickets, and reads them back to dedupe them and
to see them closed. This tracker parses, relabels and closes the same files.
Each side's own tests use its own fixtures; this one runs both real
implementations against one file, so a format change on either side is
caught here.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from ci_fleet.ledger import issues as ledger

from charlie_work.local_issue_files import scan_issues
from charlie_work.local_issues import LocalFileGitHub

REPO = "local/widget"
# A parametrized test id: backslashes and quotes that a hand-rolled writer mangles.
TITLE = 'nightly-break: tests/test_a.py::test_x[C:\\tmp\\new "q"] (break 7)'
LABELS = ["nightly-break", "model:opus", "priority:critical"]


@pytest.fixture
def filed(tmp_path: Path):
    root = tmp_path / "repo"
    issues_dir = root / "docs" / "issues"
    issues_dir.mkdir(parents=True)
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "fleet.json").write_text(
        json.dumps({"repos": {REPO: {"repo_root": str(root)}}}), encoding="utf-8"
    )
    (fleet / "config.yaml").write_text("labels:\n  ready: automated-ready\n", encoding="utf-8")
    target = ledger.resolve_target(REPO, fleet)
    ref = ledger.file_issue(
        target, title=TITLE, body="body", gh=None, labels=LABELS, today=date(2026, 10, 4)
    )
    tracker = LocalFileGitHub(repo_root=root, issues_dir=issues_dir)
    return issues_dir, fleet, target, ref, tracker


def test_the_tracker_reads_what_the_ledger_files(filed) -> None:
    issues_dir, _, _, ref, _ = filed
    scan = scan_issues(issues_dir)
    assert not scan.problems
    [issue] = scan.issues
    assert ref == f"{REPO}#001" and issue.number == 1
    assert issue.title == TITLE
    assert tuple(issue.labels) == (*LABELS, "automated-ready")
    assert issue.is_open


def test_the_ledger_reads_the_tracker_s_label_edits(filed) -> None:
    _, _, target, ref, tracker = filed
    tracker.remove_issue_label(1, "automated-ready")
    tracker.add_issue_label(1, "in-progress")
    [row] = ledger.local_issues(target)
    assert row["title"] == TITLE and row["state"] == "open"
    assert row["labels"] == [*LABELS, "in-progress"]
    assert ledger.find_open_issue(target, "(break 7)", None) == ref


def test_the_ledger_sees_the_tracker_close_it(filed) -> None:
    _, fleet, target, ref, tracker = filed
    tracker.close_issue(1)
    state = ledger.issue_state(ref, gh=None, fleet_dir=fleet)
    assert state.state == "closed" and state.close_reason == "completed"
    assert state.closed_at is not None and state.closed_at.endswith("T00:00:00.000000Z")
    assert ledger.local_issues(target)[0]["state"] == "closed"
    assert ledger.find_open_issue(target, "(break 7)", None) is None
