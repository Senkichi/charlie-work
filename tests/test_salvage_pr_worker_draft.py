"""``_open_salvage_pr`` preferring a worker's own drafted PR title/body (cw#1771).

Workers no longer attempt ``gh pr create`` themselves (see
``prompts/worker_sections/push_pr_outcome.md``): they draft the PR title and
body into ``.worker-outcome.json`` and stop, and the orchestrator opens the
PR from that draft. These tests pin the preference logic in
``_open_salvage_pr`` -- drafted content wins verbatim when present and
non-empty, the existing "Salvaged work for #N" synthesis is the fallback for
a worker that never wrote one (the genuine crash case), and the drafted body
still goes through the same closing-reference validation as a synthesized
one.
"""

from __future__ import annotations

from pathlib import Path

from _helpers import _init_git_repo
from _salvage_fixtures import _SalvageTestGitHub, _salvage_labels
from _worktree_fixtures import _git

from charlie_work.config import OrchestratorConfig
from charlie_work.workflow import _open_salvage_pr


def _commit_file(repo_root: Path, path: str, content: str, message: str) -> None:
    target = repo_root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _git(repo_root, "add", path)
    _git(repo_root, "commit", "-m", message)


def _make_branch(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    _git(repo_root, "checkout", "-b", "agent/issue-777")
    _commit_file(repo_root, "src/thing.py", "x = 1\n", "fix: thing")
    return repo_root


def _open(repo_root: Path, *, worker_outcome: dict | None, issue_number: int = 777):
    config = OrchestratorConfig()
    active_labels, issue_labels = _salvage_labels(config)
    gh = _SalvageTestGitHub(repo_root=repo_root)

    pr_number, error, _closing_ref = _open_salvage_pr(
        gh=gh,
        config=config,
        repo_root=repo_root,
        branch="agent/issue-777",
        base_ref="main",
        issue_number=issue_number,
        active_labels=active_labels,
        issue_labels=issue_labels,
        issue_title="Fix the thing",
        source_description="worker branch",
        worker_outcome=worker_outcome,
    )
    return pr_number, error, gh


def test_drafted_title_and_body_are_used_verbatim(tmp_path: Path) -> None:
    repo_root = _make_branch(tmp_path)
    drafted_body = (
        "Closes #777\n\nRan `pytest tests/test_thing.py` -- 3 passed.\nNo risks identified."
    )
    outcome = {
        "push_succeeded": True,
        "pr_created": False,
        "pr_title": "fix(thing): correct the off-by-one",
        "pr_body": drafted_body,
    }

    pr_number, error, gh = _open(repo_root, worker_outcome=outcome)

    assert pr_number is not None
    assert error is None
    created = gh.prs_created[0]
    assert created["title"] == "fix(thing): correct the off-by-one"
    # The closing reference is already correct, so validation must not alter it.
    assert created["body"] == drafted_body
    # And the fallback synthesis text must be absent -- proof the draft won, not
    # a coincidence of both paths producing similar text.
    assert "Salvaged by the orchestrator" not in created["body"]
    assert "Salvaged work for" not in created["title"]


def test_absent_worker_outcome_falls_back_to_synthesis(tmp_path: Path) -> None:
    repo_root = _make_branch(tmp_path)

    pr_number, error, gh = _open(repo_root, worker_outcome=None)

    assert pr_number is not None
    assert error is None
    created = gh.prs_created[0]
    assert created["title"] == "Salvaged work for #777: Fix the thing"
    assert "Salvaged by the orchestrator from a worker branch." in created["body"]


def test_worker_outcome_without_drafted_fields_falls_back_to_synthesis(tmp_path: Path) -> None:
    """The pre-#1771 outcome shape (push/PR bookkeeping only) must not crash the
    preference lookup and must still fall back cleanly."""
    repo_root = _make_branch(tmp_path)
    outcome = {"push_succeeded": True, "pr_created": False}

    pr_number, error, gh = _open(repo_root, worker_outcome=outcome)

    assert pr_number is not None
    assert error is None
    created = gh.prs_created[0]
    assert created["title"] == "Salvaged work for #777: Fix the thing"
    assert "Salvaged by the orchestrator from a worker branch." in created["body"]


def test_blank_drafted_fields_fall_back_to_synthesis(tmp_path: Path) -> None:
    """A worker that writes empty-string placeholders (rather than omitting the
    keys) must not win -- an empty PR body is worse than the boilerplate."""
    repo_root = _make_branch(tmp_path)
    outcome = {
        "push_succeeded": True,
        "pr_created": False,
        "pr_title": "   ",
        "pr_body": "",
    }

    pr_number, error, gh = _open(repo_root, worker_outcome=outcome)

    assert pr_number is not None
    assert error is None
    created = gh.prs_created[0]
    assert created["title"] == "Salvaged work for #777: Fix the thing"
    assert "Salvaged by the orchestrator from a worker branch." in created["body"]


def test_non_string_drafted_fields_fall_back_to_synthesis(tmp_path: Path) -> None:
    """A malformed outcome file (wrong JSON types) must degrade to the fallback,
    never raise and never be coerced into the created PR."""
    repo_root = _make_branch(tmp_path)
    outcome = {
        "push_succeeded": True,
        "pr_created": False,
        "pr_title": 12345,
        "pr_body": None,
    }

    pr_number, error, gh = _open(repo_root, worker_outcome=outcome)

    assert pr_number is not None
    assert error is None
    created = gh.prs_created[0]
    assert created["title"] == "Salvaged work for #777: Fix the thing"


def test_drafted_body_missing_closing_reference_still_gets_one(tmp_path: Path) -> None:
    """The drafted body goes through the same closing-reference validation as a
    synthesized one -- a worker that forgot ``Closes #N`` must not ship a PR
    the janitor's auto-close linking silently misses."""
    repo_root = _make_branch(tmp_path)
    outcome = {
        "push_succeeded": True,
        "pr_created": False,
        "pr_title": "fix(thing): correct the off-by-one",
        "pr_body": "Ran the targeted suite; all green. No risks identified.",
    }

    pr_number, error, gh = _open(repo_root, worker_outcome=outcome)

    assert pr_number is not None
    assert error is None
    created = gh.prs_created[0]
    assert "Closes #777" in created["body"]
    assert "Ran the targeted suite; all green." in created["body"]
