"""Issue #2094: a no-remote rework worker that dies without committing must re-arm.

Drives ``park_unpublishable_work`` (the single park edge) against a real git
repo and a real ``LocalFileGitHub``: the parked head is compared with the local
PR record's ``reviewed_head_sha``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from charlie_work.config import LabelConfig, OrchestratorConfig
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.local_work_park import park_unpublishable_work
from charlie_work.state import PASSIVE_OPEN_STATUS, load_state, save_state
from charlie_work.write_gate import WriteGate

BRANCH = "agent/issue-7-local"
_GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [*_GIT, *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "--initial-branch=main")
    _git(tmp_path, "config", "core.longpaths", "true")
    _git(tmp_path, "commit", "--allow-empty", "-m", "chore: seed")
    _git(tmp_path, "branch", BRANCH)
    return tmp_path


def _setup(repo: Path, *, record: dict[str, Any], issue_extra: dict[str, Any] | None = None):
    labels = LabelConfig()
    issues_dir = repo / "docs" / "issues"
    issues_dir.mkdir(parents=True)
    (issues_dir / "007_2026-09-17_x.md").write_text(
        "---\n"
        'title: "T"\n'
        "state: open\n"
        f"labels: [{labels.ready}, {labels.in_progress}]\n"
        'created: "2026-09-17"\n'
        'author: "test"\n'
        "---\nBody\n",
        encoding="utf-8",
    )
    gh = LocalFileGitHub(repo_root=repo, issues_dir=issues_dir)
    state_file = repo / "state.json"
    save_state(
        state_file,
        {
            "issues": {"7": {"number": 7, "status": "dispatched", **(issue_extra or {})}},
            "prs": {"7": {"number": 7, "local": True, "issue_number": 7, **record}},
        },
    )
    return gh, state_file, WriteGate(dry_run=False, state_path=state_file, repo="r")


def _park(gh, repo, config, write_gate, failure_kind: str | None = None):
    return park_unpublishable_work(
        gh, config, repo, BRANCH, 7, {LabelConfig().in_progress}, failure_kind, write_gate
    )


def _labels(gh) -> set[str]:
    return {entry["name"] for entry in gh.issue_view(7)["labels"]}


@pytest.mark.parametrize(
    ("decision", "status"),
    [("request_changes", "request_changes"), ("approved", "rework_requested")],
)
def test_no_commit_exit_rearms_rework(repo: Path, decision: str, status: str) -> None:
    head = _git(repo, "rev-parse", BRANCH)
    gh, state_file, gate = _setup(
        repo, record={"decision": decision, "status": status, "reviewed_head_sha": head}
    )

    assert _park(gh, repo, OrchestratorConfig(), gate) == (True, None)

    state = load_state(state_file)
    assert state["issues"]["7"]["status"] == "rework_requested"
    assert state["prs"]["7"]["status"] == "rework_requested"
    assert state["prs"]["7"]["no_op_rework_attempts"] == 1
    labels = LabelConfig()
    assert labels.needs_rework in _labels(gh)
    assert labels.review_ready not in _labels(gh)
    assert [e for e in state["events"] if e["kind"] == "local_no_op_rework_rearmed"]
    assert not [e for e in state["events"] if e["kind"] == "local_work_ready"]


def test_committed_worker_still_parks_review_ready(repo: Path) -> None:
    reviewed = _git(repo, "rev-parse", BRANCH)
    _git(
        repo,
        "update-ref",
        f"refs/heads/{BRANCH}",
        _git(repo, "commit-tree", "HEAD^{tree}", "-p", reviewed, "-m", "fix: work"),
    )
    gh, state_file, gate = _setup(
        repo,
        record={
            "decision": "request_changes",
            "status": "request_changes",
            "reviewed_head_sha": reviewed,
        },
    )

    assert _park(gh, repo, OrchestratorConfig(), gate) == (True, None)

    state = load_state(state_file)
    assert state["issues"]["7"]["status"] == PASSIVE_OPEN_STATUS
    assert LabelConfig().review_ready in _labels(gh)


def test_repeat_is_bounded_then_escalates(repo: Path) -> None:
    """A classified non-throttle death still counts and escalates at the cap."""
    failure_kind = "stalled"
    head = _git(repo, "rev-parse", BRANCH)
    config = OrchestratorConfig()
    gh, state_file, gate = _setup(
        repo,
        record={
            "decision": "request_changes",
            "status": "request_changes",
            "reviewed_head_sha": head,
            "no_op_rework_attempts": config.review.max_no_op_rework_attempts,
        },
    )

    assert _park(gh, repo, config, gate, failure_kind) == (True, None)

    state = load_state(state_file)
    issue = state["issues"]["7"]
    assert issue["status"] == "escalated"
    assert issue["escalation_reason"] == "no_op_rework_attempts_cap_exceeded"
    labels = LabelConfig()
    assert labels.operator_queue in _labels(gh)
    assert labels.review_ready not in _labels(gh)


def _rearm_events(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        e.get("payload", e) for e in state["events"] if e["kind"] == "local_no_op_rework_rearmed"
    ]


@pytest.mark.parametrize(
    ("passed_kind", "stamped_kind"),
    [
        ("rate_limited", None),
        ("quota_exhausted", None),
        # The park edge got no kind, but the classifier stamped this epoch's
        # entry: the locked stamp is the classification.
        (None, "rate_limited"),
    ],
)
def test_provider_throttle_death_rearms_without_counting(
    repo: Path, passed_kind: str | None, stamped_kind: str | None
) -> None:
    """mdls #109 (2026-10-06): three throttled deaths escalated an issue no
    worker ever attempted. A throttle death re-arms, never counts, never
    escalates -- even with the counter already at the cap."""
    head = _git(repo, "rev-parse", BRANCH)
    config = OrchestratorConfig()
    cap = config.review.max_no_op_rework_attempts
    gh, state_file, gate = _setup(
        repo,
        record={
            "decision": "request_changes",
            "status": "request_changes",
            "reviewed_head_sha": head,
            "no_op_rework_attempts": cap,
        },
        issue_extra=(
            {"dead_worker_failure_kind": stamped_kind} if stamped_kind is not None else None
        ),
    )

    assert _park(gh, repo, config, gate, passed_kind) == (True, None)

    state = load_state(state_file)
    assert state["issues"]["7"]["status"] == "rework_requested"
    assert state["prs"]["7"]["status"] == "rework_requested"
    assert state["prs"]["7"]["no_op_rework_attempts"] == cap
    labels = LabelConfig()
    assert labels.needs_rework in _labels(gh)
    assert labels.operator_queue not in _labels(gh)
    [event] = _rearm_events(state)
    assert event["counted"] is False
    assert event["escalated"] is False
    assert event["reason"] == "provider_throttle"
    assert event["failure_kind"] == (passed_kind or stamped_kind)
    assert event["attempts"] == cap


def test_non_throttle_death_is_counted_in_the_event(repo: Path) -> None:
    head = _git(repo, "rev-parse", BRANCH)
    gh, state_file, gate = _setup(
        repo,
        record={
            "decision": "request_changes",
            "status": "request_changes",
            "reviewed_head_sha": head,
        },
    )

    assert _park(gh, repo, OrchestratorConfig(), gate, "stalled") == (True, None)

    [event] = _rearm_events(load_state(state_file))
    assert event["counted"] is True
    assert event["reason"] is None
    assert event["attempts"] == 1
