from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LIST_LIMIT = 500

# Module-level constants for gh --json field lists.
# These are the single source of truth for all JSON field queries to GitHub.
# All call sites must use these constants — no inline field-list literals.
ISSUE_LIST_FIELDS = "number,title,url,body,labels,assignees,author,createdAt,updatedAt,state"
ISSUE_VIEW_FIELDS = (
    "number,title,url,body,labels,assignees,author,comments,createdAt,updatedAt,state"
)
PR_LIST_FIELDS = "number,title,url,headRefName,baseRefName,body,isDraft,labels,author,updatedAt,reviewDecision,statusCheckRollup,headRefOid,isCrossRepository,mergeStateStatus,state"
PR_VIEW_FIELDS = "number,title,url,headRefName,baseRefName,body,isDraft,labels,author,updatedAt,reviewDecision,statusCheckRollup,state,mergeable,additions,deletions,headRefOid,isCrossRepository,mergeStateStatus"
PR_CHECKS_FIELDS = "name,state,bucket,link"
LABEL_LIST_FIELDS = "name"
# Minimal field lists for drift detection (reconcile.py)
RECONCILE_PR_FIELDS = (
    "number,title,url,headRefName,baseRefName,body,state,labels,isCrossRepository"
)
RECONCILE_ISSUE_FIELDS = "number,title,url,body,labels"


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

    def _list_json(self, args: list[str], *, limit: int, kind: str) -> list[dict[str, Any]]:
        result = self.run(args, json_output=True)
        items = result if isinstance(result, list) else []
        if len(items) >= limit:
            logger.warning(
                "GitHub returned %d %s, matching the page limit (%d); "
                "further items may be truncated",
                len(items),
                kind,
                limit,
            )
        return items

    def issue_list(self, ready_label: str) -> list[dict[str, Any]]:
        return self._list_json(
            [
                "issue",
                "list",
                "--state",
                "open",
                "--label",
                ready_label,
                "--limit",
                str(_LIST_LIMIT),
                "--json",
                ISSUE_LIST_FIELDS,
            ],
            limit=_LIST_LIMIT,
            kind=f"ready-labeled open issues (label={ready_label})",
        )

    def issue_view(self, number: int) -> dict[str, Any]:
        result = self.run(
            [
                "issue",
                "view",
                str(number),
                "--json",
                ISSUE_VIEW_FIELDS,
            ],
            json_output=True,
        )
        return result if isinstance(result, dict) else {}

    def pr_list(self) -> list[dict[str, Any]]:
        return self._list_json(
            [
                "pr",
                "list",
                "--state",
                "open",
                "--limit",
                str(_LIST_LIMIT),
                "--json",
                PR_LIST_FIELDS,
            ],
            limit=_LIST_LIMIT,
            kind="open PRs",
        )

    def pr_view(self, number: int) -> dict[str, Any]:
        result = self.run(
            [
                "pr",
                "view",
                str(number),
                "--json",
                PR_VIEW_FIELDS,
            ],
            json_output=True,
        )
        return result if isinstance(result, dict) else {}

    def pr_diff(self, number: int) -> str:
        result = self.run(["pr", "diff", str(number)], allow_failure=True)
        return result if isinstance(result, str) else ""

    def pr_checks(self, number: int) -> list[dict[str, Any]]:
        result = self.run(
            ["pr", "checks", str(number), "--json", PR_CHECKS_FIELDS],
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
        result = self.run(
            ["label", "list", "--limit", "200", "--json", LABEL_LIST_FIELDS], json_output=True
        )
        return result if isinstance(result, list) else []

    def label_create(self, label: str, color: str, description: str) -> None:
        # --force makes this update-or-create: bootstrap must be idempotent, and
        # without it `gh label create` errors on a pre-existing label and the
        # colour/description drift silently. `--force` is a mutation but stays
        # read-only-safe under dry-run via `_is_mutating`.
        self.run(
            ["label", "create", label, "--force", "--color", color, "--description", description],
            allow_failure=True,
        )

    def merge_pr(self, number: int, strategy: str, admin: bool = False) -> str:
        args = ["pr", "merge", str(number)]
        if admin:
            args.append("--admin")
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

    def pr_update_branch(self, pr_number: int) -> bool:
        """Update a PR's branch with the latest changes from its base.

        Uses `gh pr update-branch`. Returns True on success, False on failure
        (conflicts, network errors, etc.). Never raises — per-PR failures are
        reported as values and must not abort a batch operation.
        """
        try:
            self.run(["pr", "update-branch", str(pr_number)])
            return True
        except GitHubError:
            return False

    def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
        """Check which of the given issue numbers are currently open.

        Returns a set of issue numbers that are open. Issues that don't exist
        or are closed are not included in the result. This is used for the
        dependency gate to check if blocker issues are still open.

        Args:
            issue_numbers: List of issue numbers to check

        Returns:
            Set of issue numbers that are currently open
        """
        if not issue_numbers:
            return set()

        open_issues: set[int] = set()
        for number in issue_numbers:
            try:
                issue = self.issue_view(number)
                # GitHub API returns issues regardless of state; we need to check
                # the state field. If the issue doesn't exist, issue_view raises.
                if issue.get("state") == "open":
                    open_issues.add(number)
            except (GitHubError, ValueError, TypeError):
                # Issue doesn't exist or API error — treat as not blocking
                continue

        return open_issues


def label_names(item: dict[str, Any]) -> set[str]:
    labels = item.get("labels") or []
    names: set[str] = set()
    for label in labels:
        if isinstance(label, dict) and label.get("name"):
            names.add(str(label["name"]))
        elif isinstance(label, str):
            names.add(label)
    return names


# GitHub's own issue-closing keyword set, used here to decide whether a `#N`
# reference in freeform text actually links the PR to issue N.
_CLOSING_KEYWORD_REF = re.compile(
    r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)", flags=re.IGNORECASE
)
# The orchestrator's own branch convention (agent/issue-N-slug). A head ref is
# the trusted signal because the orchestrator created it at dispatch.
_BRANCH_ISSUE_REF = re.compile(r"issue[-_/](\d+)", flags=re.IGNORECASE)


def linked_issue_number(
    pr: dict[str, Any],
    *,
    is_cross_repository: bool | None,
    branch_prefix: str,
) -> int | None:
    """Resolve the issue a PR is bound to, safe against hijack.

    A bare ``#N`` substring in an attacker-controlled PR *title* must never
    bind the PR to issue N — that let any external PR author drive another
    issue's label/merge transitions. So: trust the head ref only when the PR
    is same-repo (isCrossRepository == false) AND the branch starts with the
    configured ``branch_prefix``. For fork PRs, never bind for lifecycle
    purposes — return None before any keyword scan. (GitHub's own auto-close
    on merge is GitHub's policy for issue state; the orchestrator's label
    lifecycle is ours.)

    When is_cross_repository is None (provenance unknown), treat as
    cross-repo for trust purposes — bind nothing via branch name or closing
    keyword (fail closed).
    """
    # Cross-repo PRs or unknown provenance never bind for lifecycle purposes
    if is_cross_repository is True or is_cross_repository is None:
        return None

    head = str(pr.get("headRefName") or "")
    match = _BRANCH_ISSUE_REF.search(head)
    if match:
        # Only trust the branch ref when:
        # 1. PR is same-repo (is_cross_repository is not True)
        # 2. Branch starts with the configured prefix
        has_correct_prefix = head.startswith(branch_prefix)
        if has_correct_prefix:
            return int(match.group(1))
    # For same-repo PRs, trust closing keywords in title/body
    for text in (str(pr.get("title") or ""), str(pr.get("body") or "")):
        match = _CLOSING_KEYWORD_REF.search(text)
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


# Blocker declaration patterns for dependency gate
# Case-insensitive patterns: "Blocked by #N", "Depends on #N", "Blocked-by: #N"
# Handles comma-separated lists like "Blocked by #743, #744"
_BLOCKER_PATTERNS = [
    re.compile(r"blocked\s+by\s+#\d+(?:\s*,\s*#\d+)*", flags=re.IGNORECASE),
    re.compile(r"depends\s+on\s+#\d+(?:\s*,\s*#\d+)*", flags=re.IGNORECASE),
    re.compile(r"blocked-by:\s*#\d+(?:\s*,\s*#\d+)*", flags=re.IGNORECASE),
]


def parse_blockers(text: str) -> list[int]:
    """Parse blocker issue numbers from issue body text.

    Returns a list of issue numbers declared as blockers using patterns like:
    - "Blocked by #N"
    - "Depends on #N"
    - "Blocked-by: #N"

    Handles comma-separated lists (e.g., "Blocked by #743, #744").
    Returns an empty list if no blockers are found.
    """
    if not text:
        return []

    blockers: set[int] = set()
    # Check if they appear in blocker context
    for pattern in _BLOCKER_PATTERNS:
        for match in pattern.finditer(text):
            # Extract the full match and find all #N references within it
            match_text = match.group(0)
            numbers_in_match = re.findall(r"#(\d+)", match_text)
            for num_str in numbers_in_match:
                try:
                    blockers.add(int(num_str))
                except (ValueError, TypeError):
                    # Skip malformed numbers
                    continue

    return sorted(blockers)


def get_github_issue_dependencies(gh: GitHub, issue_number: int) -> list[int]:
    """Fetch GitHub's native issue dependencies (blocked_by relationships).

    Uses the GitHub API to check for issue dependencies. Tolerates 404/410 errors
    for repos that don't have the feature enabled. Returns an empty list on any
    error (fail-open for compatibility).

    Args:
        gh: GitHub client instance
        issue_number: The issue number to check dependencies for

    Returns:
        List of issue numbers that block this issue via GitHub's native API
    """
    result = gh.run(
        [
            "api",
            f"repos/{{owner}}/{{repo}}/issues/{issue_number}/dependencies/blocked_by",
        ],
        json_output=True,
        allow_failure=True,
    )

    # Handle different return types from allow_failure=True
    if result is None:
        # Transient error or gh not available — fail open with warning
        logger.warning(
            f"GitHub dependencies API returned None for issue #{issue_number} - treating as no dependencies"
        )
        return []
    elif isinstance(result, dict):
        # 404/410 error response — feature not available on this repo
        # This is expected for repos without dependencies enabled
        return []
    elif isinstance(result, list):
        # Extract issue numbers from the dependency list
        return [int(dep.get("number", 0)) for dep in result if dep.get("number")]
    else:
        # Unexpected type — fail open with warning
        logger.warning(
            f"GitHub dependencies API returned unexpected type {type(result)} for issue #{issue_number} - treating as no dependencies"
        )
        return []
