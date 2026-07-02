from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class GitHubError(RuntimeError):
    pass


@dataclass(frozen=True)
class GitHub:
    repo_root: Path
    dry_run: bool = False

    def run(
        self, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> Any:
        command = ["gh", *args]
        if self.dry_run and _is_mutating(args):
            return [] if json_output else "DRY-RUN: " + " ".join(command)
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_root,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=not allow_failure,
            )
        except FileNotFoundError as exc:
            raise GitHubError("GitHub CLI `gh` is not installed or not on PATH.") from exc
        except subprocess.CalledProcessError as exc:
            raise GitHubError(exc.stderr.strip() or exc.stdout.strip() or str(exc)) from exc
        output = result.stdout.strip()
        if result.returncode != 0 and allow_failure and not output:
            return None if json_output else result.stderr.strip()
        if not json_output:
            return output
        if not output:
            return None
        try:
            return json.loads(output)
        except json.JSONDecodeError as exc:
            raise GitHubError(f"Expected JSON from gh command: {' '.join(command)}") from exc

    def issue_list(self, ready_label: str) -> list[dict[str, Any]]:
        result = self.run(
            [
                "issue",
                "list",
                "--state",
                "open",
                "--label",
                ready_label,
                "--limit",
                "200",
                "--json",
                "number,title,url,body,labels,assignees,author,createdAt,updatedAt",
            ],
            json_output=True,
        )
        return result if isinstance(result, list) else []

    def issue_view(self, number: int) -> dict[str, Any]:
        result = self.run(
            [
                "issue",
                "view",
                str(number),
                "--json",
                "number,title,url,body,labels,assignees,author,comments,createdAt,updatedAt",
            ],
            json_output=True,
        )
        return result if isinstance(result, dict) else {}

    def pr_list(self) -> list[dict[str, Any]]:
        result = self.run(
            [
                "pr",
                "list",
                "--state",
                "open",
                "--limit",
                "200",
                "--json",
                "number,title,url,headRefName,baseRefName,body,isDraft,labels,author,updatedAt,reviewDecision,statusCheckRollup",
            ],
            json_output=True,
        )
        return result if isinstance(result, list) else []

    def pr_view(self, number: int) -> dict[str, Any]:
        result = self.run(
            [
                "pr",
                "view",
                str(number),
                "--json",
                "number,title,url,headRefName,baseRefName,body,isDraft,labels,author,updatedAt,reviewDecision,statusCheckRollup,files,commits",
            ],
            json_output=True,
        )
        return result if isinstance(result, dict) else {}

    def pr_diff(self, number: int) -> str:
        result = self.run(["pr", "diff", str(number)], allow_failure=True)
        return result if isinstance(result, str) else ""

    def pr_checks(self, number: int) -> list[dict[str, Any]]:
        result = self.run(
            ["pr", "checks", str(number), "--json", "name,state,bucket,link"],
            json_output=True,
            allow_failure=True,
        )
        return result if isinstance(result, list) else []

    def add_issue_label(self, number: int, label: str) -> None:
        self.run(["issue", "edit", str(number), "--add-label", label])

    def remove_issue_label(self, number: int, label: str) -> None:
        self.run(["issue", "edit", str(number), "--remove-label", label], allow_failure=True)

    def issue_comment(self, number: int, body_file: Path) -> None:
        self.run(["issue", "comment", str(number), "--body-file", str(body_file)])

    def pr_comment(self, number: int, body_file: Path) -> None:
        self.run(["pr", "comment", str(number), "--body-file", str(body_file)])

    def label_list(self) -> list[dict[str, Any]]:
        result = self.run(["label", "list", "--limit", "200", "--json", "name"], json_output=True)
        return result if isinstance(result, list) else []

    def label_create(self, label: str, color: str, description: str) -> None:
        self.run(
            ["label", "create", label, "--color", color, "--description", description],
            allow_failure=True,
        )

    def merge_pr(self, number: int, strategy: str) -> str:
        args = ["pr", "merge", str(number)]
        if strategy == "merge":
            args.append("--merge")
        elif strategy == "rebase":
            args.append("--rebase")
        else:
            args.append("--squash")
        # Branch deletion is deliberately NOT part of this call: `gh pr merge
        # --delete-branch` also deletes/switches the LOCAL branch and fails when
        # the head branch is checked out in a worktree, which used to abort the
        # post-merge label update. Use `delete_branch` separately, best-effort.
        # `self.run` raises GitHubError on a non-zero exit, so reaching this line
        # means the merge succeeded. `gh pr merge` prints its success line to
        # stderr, leaving stdout empty — fall back to an explicit success string
        # so callers see a truthy result (otherwise `merged` reads as False on a
        # successful merge).
        output = str(self.run(args))
        return output or f"merged #{number}"

    def delete_branch(self, branch: str) -> bool:
        """Best-effort deletion of the REMOTE head branch after a merge.

        Uses the git-refs API so local checkouts and worktrees are never
        touched. Returns False instead of raising — a deletion failure must
        never abort the merge/label sequence.
        """
        try:
            self.run(["api", "-X", "DELETE", f"repos/{{owner}}/{{repo}}/git/refs/heads/{branch}"])
        except GitHubError:
            return False
        return True


def label_names(item: dict[str, Any]) -> set[str]:
    labels = item.get("labels") or []
    names: set[str] = set()
    for label in labels:
        if isinstance(label, dict) and label.get("name"):
            names.add(str(label["name"]))
        elif isinstance(label, str):
            names.add(label)
    return names


def linked_issue_number(pr: dict[str, Any]) -> int | None:
    head = str(pr.get("headRefName") or "")
    title = str(pr.get("title") or "")
    body = str(pr.get("body") or "")
    for text in (head, title):
        match = re.search(r"(?:issue[-_/]|#)(\d+)", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    match = re.search(r"(?:closes|fixes|resolves)\s+#(\d+)", body, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _is_mutating(args: list[str]) -> bool:
    if not args:
        return False
    text = " ".join(args)
    readonly_prefixes = (
        "issue list",
        "issue view",
        "pr list",
        "pr view",
        "pr diff",
        "pr checks",
        "label list",
        "auth status",
    )
    return not any(text.startswith(prefix) for prefix in readonly_prefixes)
