"""Regression tests for issue #956: deduplicate the salvage-PR path.

These target the extracted ``_open_salvage_pr`` helper and the two wrappers
that call it (``_attempt_salvage`` and ``_open_pr_for_orphaned_branch``).
"""

import json
from pathlib import Path
from typing import Any

import pytest

from charlie_work.config import OrchestratorConfig


class _SalvageTestGitHub:
    """Minimal fake for the ``GitHubLike`` surface used by the salvage helpers."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        pr_create_return: int | None = 101,
        remove_ok: bool = True,
        add_ok: bool = True,
        repo_slug: str = "owner/repo",
        closing_issue_numbers: list[int] | None = None,
        pr_view_raises: bool = False,
    ) -> None:
        self.repo_root = repo_root
        self.dry_run = False
        self.pr_create_return = pr_create_return
        self._remove_ok = remove_ok
        self._add_ok = add_ok
        self._repo_slug = repo_slug
        # None means "same as the created PR's own issue number" -- set by
        # each test via `closing_issue_numbers_override` when it needs to
        # simulate a mismatch; the default keeps existing tests (which never
        # exercise the post-create probe) unaffected.
        self._closing_issue_numbers = closing_issue_numbers
        self._pr_view_raises = pr_view_raises
        self.prs_created: list[dict[str, Any]] = []
        self.labels_removed: list[tuple[int, str]] = []
        self.labels_added: list[tuple[int, str]] = []
        self.pr_view_calls: list[int] = []

    def pr_create(self, head: str, base: str, title: str, body: str) -> int | None:
        self.prs_created.append({"head": head, "base": base, "title": title, "body": body})
        return self.pr_create_return

    def remove_issue_label(self, number: int, label: str) -> bool:
        self.labels_removed.append((number, label))
        return self._remove_ok

    def add_issue_label(self, number: int, label: str) -> bool:
        self.labels_added.append((number, label))
        return self._add_ok

    def name_with_owner(self) -> str:
        return self._repo_slug

    def pr_view(self, number: int, *, fields: str = "") -> dict[str, Any]:
        self.pr_view_calls.append(number)
        if self._pr_view_raises:
            raise RuntimeError("gh pr view unavailable")
        numbers = self._closing_issue_numbers
        if numbers is None:
            return {"closingIssuesReferences": []}
        return {"closingIssuesReferences": [{"number": n} for n in numbers]}


def _salvage_labels(config: OrchestratorConfig) -> tuple[set[str], set[str]]:
    active = {config.labels.in_progress}
    issue = {config.labels.in_progress}
    return active, issue


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
    )

    assert salvaged is True
    assert error is not None
    state = load_state(state_file)
    events = [e for e in state["events"] if e["kind"] == "session_salvaged"]
    assert events[0]["payload"]["label_write_ok"] is False
    assert events[0]["payload"]["label_error"] is not None
