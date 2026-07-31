from __future__ import annotations

from pathlib import Path
from typing import Any

from charlie_work.github import GitHubLike


class StubGitHubLike(GitHubLike):
    """Base class for GitHub test doubles.

    Implements :class:`~charlie_work.github.GitHubLike` with
    ``NotImplementedError`` stubs so a fake only overrides the methods its
    test exercises. This keeps the production contract explicit and makes an
    incomplete double a type-check error instead of a runtime
    ``AttributeError``.
    """

    def invalidate_list_cache(self) -> None:
        raise NotImplementedError

    def run(
        self, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> Any:
        raise NotImplementedError

    def pr_create(self, head: str, base: str, title: str, body: str) -> int | None:
        raise NotImplementedError

    def check_graphql_rate_limit(self, threshold: int = 1500) -> tuple[bool, int, int | None]:
        raise NotImplementedError

    def issue_list(self, labels: Any = None, state: Any = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    def issue_view(self, number: int) -> dict[str, Any]:
        raise NotImplementedError

    def pr_list(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def merged_pr_list(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def merged_prs_for_issue(self, issue_number: int, branch_prefix: str) -> Any:
        raise NotImplementedError

    def pr_view(self, number: int) -> dict[str, Any]:
        raise NotImplementedError

    def pr_diff(self, number: int) -> str:
        raise NotImplementedError

    def pr_checks(self, number: int) -> list[dict[str, Any]] | None:
        raise NotImplementedError

    def actions_job(self, job_id: int) -> dict[str, Any] | None:
        raise NotImplementedError

    def check_run_annotations(self, check_run_id: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    def commit(self, sha: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def commit_check_runs(self, sha: str) -> list[dict[str, Any]] | None:
        raise NotImplementedError

    def compare(self, base: str, head: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def compare_diff(self, base: str, head: str) -> str | None:
        raise NotImplementedError

    def add_issue_label(self, number: int, label: str) -> bool:
        raise NotImplementedError

    def remove_issue_label(self, number: int, label: str) -> bool:
        raise NotImplementedError

    def add_pr_label(self, number: int, label: str) -> bool:
        raise NotImplementedError

    def remove_pr_label(self, number: int, label: str) -> bool:
        raise NotImplementedError

    def close_issue(self, number: int) -> bool:
        raise NotImplementedError

    def pr_comment(self, number: int, body_file: Path) -> None:
        raise NotImplementedError

    def label_list(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def label_create(self, label: str, color: str, description: str) -> None:
        raise NotImplementedError

    def merge_pr(
        self, number: int, strategy: str, admin: bool = False, merge_flags: tuple[str, ...] = ()
    ) -> str:
        raise NotImplementedError

    def delete_branch(self, branch: str) -> bool:
        raise NotImplementedError

    def pr_update_branch(self, pr_number: int) -> bool:
        raise NotImplementedError

    def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
        raise NotImplementedError

    def name_with_owner(self) -> str:
        raise NotImplementedError
