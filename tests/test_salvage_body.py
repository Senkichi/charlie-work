"""Tests for ``summarize_branch_work`` (src/charlie_work/worktree.py).

Salvage PRs are opened by the orchestrator, not by the worker that did the
work, so a hardcoded boilerplate body used to be the only text on the PR.
The janitor's body gate (``_TESTS_OR_RATIONALE_RE`` in
``src/charlie_work/janitor.py``) requires the body to mention
tests/verification/rationale, so every salvage PR failed that gate on text
the orchestrator itself wrote. ``summarize_branch_work`` derives the body
from the worker's own commit log and touched test files instead.

These tests build real git repos under ``tmp_path`` (no mocking of git
plumbing) so the ``base..branch`` vs ``base...branch`` distinction and the
ref-validation failure paths are exercised against actual git behavior.
"""

from __future__ import annotations

from pathlib import Path

from test_charlie_work import _init_git_repo
from test_worktree import _git

from charlie_work.janitor import _TESTS_OR_RATIONALE_RE
from charlie_work.worktree import _SALVAGE_LOG_LIMIT, summarize_branch_work

_TEST_GLOBS = ("tests/**", "test_*.py", "*_test.py", "conftest.py")


def _commit_file(repo_root: Path, path: str, content: str, message: str) -> None:
    target = repo_root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _git(repo_root, "add", path)
    _git(repo_root, "commit", "-m", message)


def _empty_commit(repo_root: Path, message: str) -> None:
    _git(repo_root, "commit", "--allow-empty", "-m", message)


def test_branch_with_commits_lists_each_commit_subject(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    _git(repo_root, "checkout", "-b", "feature")
    _commit_file(repo_root, "src/thing.py", "x = 1\n", "feat: add thing")
    _commit_file(repo_root, "src/other.py", "y = 2\n", "fix: correct other")

    summary = summarize_branch_work(repo_root, "feature", "main", test_path_globs=_TEST_GLOBS)

    assert "feat: add thing" in summary
    assert "fix: correct other" in summary


def test_summary_appended_to_boilerplate_passes_janitor_gate_with_test_file(
    tmp_path: Path,
) -> None:
    """The rendered body, once appended to the salvage boilerplate exactly as
    ``_open_salvage_pr`` does, must satisfy the real janitor regex -- imported
    directly so drift in the gate is caught here too."""
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    _git(repo_root, "checkout", "-b", "feature")
    _commit_file(repo_root, "src/thing.py", "x = 1\n", "feat: add thing")
    _commit_file(
        repo_root,
        "tests/test_thing.py",
        "def test_thing(): pass\n",
        "test: cover thing",
    )

    summary = summarize_branch_work(repo_root, "feature", "main", test_path_globs=_TEST_GLOBS)
    assert summary

    body = f"Closes #123\n\nSalvaged by the orchestrator from a worker branch.\n\n{summary}"
    assert _TESTS_OR_RATIONALE_RE.search(body)


def test_no_commits_ahead_returns_empty_and_boilerplate_only_body_fails_gate(
    tmp_path: Path,
) -> None:
    """Critical negative case: a branch identical to base has nothing honest
    to report. ``summarize_branch_work`` must return "" rather than any
    phrasing of "no tests changed" (which would itself contain the gate's
    keywords and silently launder every no-op salvage past the gate)."""
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    _git(repo_root, "checkout", "-b", "no-commits")

    summary = summarize_branch_work(repo_root, "no-commits", "main", test_path_globs=_TEST_GLOBS)
    assert summary == ""

    body = "Closes #123\n\nSalvaged by the orchestrator from a worker branch."
    assert not _TESTS_OR_RATIONALE_RE.search(body)


def test_nonexistent_branch_returns_empty_string(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)

    summary = summarize_branch_work(
        repo_root, "does-not-exist", "main", test_path_globs=_TEST_GLOBS
    )

    assert summary == ""


def test_invalid_ref_names_return_empty_string_without_raising(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)

    assert summarize_branch_work(repo_root, "bad..name", "main", test_path_globs=_TEST_GLOBS) == ""
    assert summarize_branch_work(repo_root, "@{", "main", test_path_globs=_TEST_GLOBS) == ""
    assert (
        summarize_branch_work(repo_root, "feature", "bad..name", test_path_globs=_TEST_GLOBS) == ""
    )


def test_log_range_uses_two_dots_not_three(tmp_path: Path) -> None:
    """Regression test: ``git log base..branch`` (two dots, "on branch, not on
    base") must be used for the commit log, not ``base...branch`` (three
    dots, the symmetric difference). With three dots, commits made on the
    base branch after the fork point leak into the summary and get
    misattributed to the worker."""
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)

    _git(repo_root, "checkout", "-b", "feature")
    _commit_file(repo_root, "src/thing.py", "x = 1\n", "feat: worker's own change")

    # Base branch gains a commit of its own AFTER the fork point.
    _git(repo_root, "checkout", "main")
    _commit_file(repo_root, "src/base_only.py", "z = 3\n", "chore: base-only change")

    summary = summarize_branch_work(repo_root, "feature", "main", test_path_globs=_TEST_GLOBS)

    assert "feat: worker's own change" in summary
    assert "chore: base-only change" not in summary


def test_test_path_globs_classification(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)

    _git(repo_root, "checkout", "-b", "with-tests")
    _commit_file(
        repo_root,
        "tests/test_x.py",
        "def test_x(): pass\n",
        "test: add test_x",
    )
    with_tests_summary = summarize_branch_work(
        repo_root, "with-tests", "main", test_path_globs=_TEST_GLOBS
    )
    assert "## Tests" in with_tests_summary
    assert "tests/test_x.py" in with_tests_summary
    assert "changed no test files" not in with_tests_summary

    _git(repo_root, "checkout", "main")
    _git(repo_root, "checkout", "-b", "product-only")
    _commit_file(repo_root, "src/product.py", "p = 1\n", "feat: product only")
    product_only_summary = summarize_branch_work(
        repo_root, "product-only", "main", test_path_globs=_TEST_GLOBS
    )
    assert "changed no test files" in product_only_summary


def test_salvage_log_limit_elides_and_reports_remainder(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    _git(repo_root, "checkout", "-b", "many-commits")

    total = _SALVAGE_LOG_LIMIT + 2
    for i in range(total):
        _empty_commit(repo_root, f"chore: commit number {i}")

    summary = summarize_branch_work(repo_root, "many-commits", "main", test_path_globs=_TEST_GLOBS)
    summary_lines = set(summary.splitlines())

    # git log lists newest first, so the *first* _SALVAGE_LOG_LIMIT subjects
    # shown are the most recent commits: numbers (total - 1) down to
    # (total - _SALVAGE_LOG_LIMIT). Match whole lines -- "commit number 1" is
    # a substring of "commit number 19", so a plain substring check would
    # give false positives for the elided commits.
    for i in range(total - _SALVAGE_LOG_LIMIT, total):
        assert f"- chore: commit number {i}" in summary_lines
    for i in range(0, total - _SALVAGE_LOG_LIMIT):
        assert f"- chore: commit number {i}" not in summary_lines

    assert f"... and {total - _SALVAGE_LOG_LIMIT} more commit(s)" in summary
