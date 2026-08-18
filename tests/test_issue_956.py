"""Regression tests for issue #956: deduplicate the salvage-PR path.

These target the extracted ``_open_salvage_pr`` helper and the two wrappers
that call it (``_attempt_salvage`` and ``_open_pr_for_orphaned_branch``).
"""

import json
from pathlib import Path

import pytest

from _salvage_fixtures import _SalvageTestGitHub, _salvage_labels
from charlie_work.config import OrchestratorConfig
from charlie_work.write_gate import WriteGate


def _wg(state_file: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo="charlie-work")


def test_open_salvage_pr_creates_pr_and_moves_labels(tmp_path: Path) -> None:
    """The helper opens a PR, derives the title from the issue title, and swaps labels."""
    from charlie_work.workflow import _open_salvage_pr

    config = OrchestratorConfig()
    active_labels, issue_labels = _salvage_labels(config)
    gh = _SalvageTestGitHub(repo_root=tmp_path)

    pr_number, error, _closing_ref = _open_salvage_pr(
        gh=gh,
        config=config,
        repo_root=tmp_path,
        branch="agent/issue-956",
        base_ref="main",
        issue_number=956,
        active_labels=active_labels,
        issue_labels=issue_labels,
        issue_title="Deduplicate salvage PR path",
        source_description="completed-but-unpublished worker worktree",
    )

    assert pr_number == 101
    assert error is None
    assert len(gh.prs_created) == 1
    created = gh.prs_created[0]
    assert created["title"] == "Salvaged work for #956: Deduplicate salvage PR path"
    assert created["base"] == "main"
    assert created["head"] == "agent/issue-956"
    assert "Closes #956" in created["body"]
    assert "completed-but-unpublished worker worktree" in created["body"]
    assert (956, config.labels.in_progress) in gh.labels_removed
    assert (956, config.labels.pr_open) in gh.labels_added


def test_open_salvage_pr_falls_back_to_fixed_title(tmp_path: Path) -> None:
    """When no issue title is supplied, the helper keeps the historical title pattern."""
    from charlie_work.workflow import _open_salvage_pr

    config = OrchestratorConfig()
    active_labels, issue_labels = _salvage_labels(config)
    gh = _SalvageTestGitHub(repo_root=tmp_path)

    _open_salvage_pr(
        gh=gh,
        config=config,
        repo_root=tmp_path,
        branch="agent/issue-956",
        base_ref="main",
        issue_number=956,
        active_labels=active_labels,
        issue_labels=issue_labels,
    )

    assert gh.prs_created[0]["title"] == "Salvaged work for issue #956"


def test_open_salvage_pr_returns_none_on_pr_create_failure(tmp_path: Path) -> None:
    """A failed ``pr_create`` must return no PR number and must not touch labels."""
    from charlie_work.workflow import _open_salvage_pr

    config = OrchestratorConfig()
    active_labels, issue_labels = _salvage_labels(config)
    gh = _SalvageTestGitHub(repo_root=tmp_path, pr_create_return=None)

    pr_number, error, _closing_ref = _open_salvage_pr(
        gh=gh,
        config=config,
        repo_root=tmp_path,
        branch="agent/issue-956",
        base_ref="main",
        issue_number=956,
        active_labels=active_labels,
        issue_labels=issue_labels,
    )

    assert pr_number is None
    assert "gh pr create failed" in (error or "")
    assert not gh.labels_removed
    assert not gh.labels_added


def test_open_salvage_pr_returns_pr_and_error_on_label_write_failure(tmp_path: Path) -> None:
    """If the PR is created but a label write fails, the helper still reports the PR."""
    from charlie_work.workflow import _open_salvage_pr

    config = OrchestratorConfig()
    active_labels, issue_labels = _salvage_labels(config)
    gh = _SalvageTestGitHub(repo_root=tmp_path, remove_ok=False)

    pr_number, error, _closing_ref = _open_salvage_pr(
        gh=gh,
        config=config,
        repo_root=tmp_path,
        branch="agent/issue-956",
        base_ref="main",
        issue_number=956,
        active_labels=active_labels,
        issue_labels=issue_labels,
    )

    assert pr_number == 101
    assert error is not None
    assert "label" in error.lower()
    assert (956, config.labels.in_progress) in gh.labels_removed


def test_open_salvage_pr_refuses_missing_repo_root() -> None:
    """A missing ``repo_root`` must be reported as a value, never passed to git/gh."""
    from charlie_work.workflow import _open_salvage_pr

    config = OrchestratorConfig()
    active_labels, issue_labels = _salvage_labels(config)
    gh = _SalvageTestGitHub()

    pr_number, error, _closing_ref = _open_salvage_pr(
        gh=gh,
        config=config,
        repo_root=None,
        branch="agent/issue-956",
        base_ref="main",
        issue_number=956,
        active_labels=active_labels,
        issue_labels=issue_labels,
    )

    assert pr_number is None
    assert "repo_root" in (error or "").lower()
    assert not gh.prs_created


def test_open_pr_for_orphaned_branch_refuses_missing_repo_root() -> None:
    """The orphan-branch wrapper must accept ``repo_root=None`` and guard the call."""
    from charlie_work.workflow import _open_pr_for_orphaned_branch

    config = OrchestratorConfig()
    active_labels, issue_labels = _salvage_labels(config)
    gh = _SalvageTestGitHub()

    pr_number, error, _closing_ref = _open_pr_for_orphaned_branch(
        gh=gh,
        config=config,
        repo_root=None,
        branch="agent/issue-956",
        base_ref="main",
        issue_number=956,
        active_labels=active_labels,
        issue_labels=issue_labels,
        issue_title="Orphan branch",
    )

    assert pr_number is None
    assert "repo_root" in (error or "").lower()


def test_open_pr_for_orphaned_branch_uses_issue_title(tmp_path: Path) -> None:
    """The orphan-branch wrapper must derive the PR title from the issue title."""
    from charlie_work.workflow import _open_pr_for_orphaned_branch

    config = OrchestratorConfig()
    active_labels, issue_labels = _salvage_labels(config)
    gh = _SalvageTestGitHub(repo_root=tmp_path)

    pr_number, error, _closing_ref = _open_pr_for_orphaned_branch(
        gh=gh,
        config=config,
        repo_root=tmp_path,
        branch="agent/issue-2",
        base_ref="main",
        issue_number=2,
        active_labels=active_labels,
        issue_labels=issue_labels,
        issue_title="Worker could not open the PR",
    )

    assert pr_number == 101
    assert error is None
    assert gh.prs_created[0]["title"] == "Salvaged work for #2: Worker could not open the PR"
    assert "worker branch that could not open a PR" in gh.prs_created[0]["body"]


def test_attempt_salvage_records_salvaged_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The salvage wrapper records a ``session_salvaged`` event on success."""
    from charlie_work.state import load_state
    from charlie_work.workflow import _attempt_salvage

    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": []}), encoding="utf-8")
    config = OrchestratorConfig()
    active_labels, issue_labels = _salvage_labels(config)
    gh = _SalvageTestGitHub(repo_root=tmp_path)
    monkeypatch.setattr("charlie_work.workflow.push_branch", lambda *a, **k: (True, None))

    salvaged, error = _attempt_salvage(
        gh=gh,
        config=config,
        repo_root=tmp_path,
        worktree_path=tmp_path,
        branch="agent/issue-3",
        base_ref="main",
        issue_number=3,
        active_labels=active_labels,
        issue_labels=issue_labels,
        state_file=state_file,
        failure_kind="unpublished_work",
        issue_title="Completed but unpublished",
        write_gate=_wg(state_file),
    )

    assert salvaged is True
    assert error is None
    state = load_state(state_file)
    events = [e for e in state["events"] if e["kind"] == "session_salvaged"]
    assert len(events) == 1
    assert events[0]["payload"]["issue_number"] == 3
    assert events[0]["payload"]["pr_number"] == 101
    assert events[0]["payload"]["label_write_ok"] is True


def test_attempt_salvage_records_label_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed label swap after a successful PR create is still reported as an event."""
    from charlie_work.state import load_state
    from charlie_work.workflow import _attempt_salvage

    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": []}), encoding="utf-8")
    config = OrchestratorConfig()
    active_labels, issue_labels = _salvage_labels(config)
    gh = _SalvageTestGitHub(repo_root=tmp_path, add_ok=False)
    monkeypatch.setattr("charlie_work.workflow.push_branch", lambda *a, **k: (True, None))

    salvaged, error = _attempt_salvage(
        gh=gh,
        config=config,
        repo_root=tmp_path,
        worktree_path=tmp_path,
        branch="agent/issue-4",
        base_ref="main",
        issue_number=4,
        active_labels=active_labels,
        issue_labels=issue_labels,
        state_file=state_file,
        failure_kind="unpublished_work",
        issue_title="Completed but labels fail",
        write_gate=_wg(state_file),
    )

    assert salvaged is True
    assert error is not None
    state = load_state(state_file)
    events = [e for e in state["events"] if e["kind"] == "session_salvaged"]
    assert events[0]["payload"]["label_write_ok"] is False
    assert events[0]["payload"]["label_error"] is not None
