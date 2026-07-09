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
PR_CHECKS_FIELDS = "name,state,bucket,link,databaseId"
LABEL_LIST_FIELDS = "name"
# Minimal field lists for drift detection (reconcile.py)
RECONCILE_PR_FIELDS = (
    "number,title,url,headRefName,baseRefName,body,state,labels,isCrossRepository"
)
RECONCILE_ISSUE_FIELDS = "number,title,url,body,labels"

# Flag constants for merge_pr — single source of truth for both argv construction
# and config validation. Derive ORCHESTRATOR_MANAGED_MERGE_FLAGS from these so that
# adding a new orchestrator-managed flag to merge_pr automatically rejects it in
# config validation (prevents drift issue #107).
_STRATEGY_FLAGS = {"merge": "--merge", "squash": "--squash", "rebase": "--rebase"}
_DELETE_BRANCH_FLAG = "--delete-branch"
_ADMIN_FLAG = "--admin"
ORCHESTRATOR_MANAGED_MERGE_FLAGS: frozenset[str] = frozenset(
    {*_STRATEGY_FLAGS.values(), _DELETE_BRANCH_FLAG, _ADMIN_FLAG}
)


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

    def _run_bool(self, args: list[str]) -> bool:
        """Run a gh command and return True iff returncode == 0.

        This is a private helper for label operations that need boolean success
        semantics without inferring from stdout/stderr string shape. Never raises
        — failures are returned as False (allow_failure semantics). Dry-run mode
        returns True (the operation would succeed if not for dry-run).
        """
        command = ["gh", *args]
        if self.dry_run and _is_mutating(args):
            return True
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_root,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            return False
        return result.returncode == 0

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

    def issue_list(self, labels=None, state=None) -> list[dict[str, Any]]:
        # Support both old signature (ready_label: str) and new (labels=None, state=None)
        if isinstance(labels, str):
            ready_label = labels
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

        # New signature: labels as list, state as string
        args = [
            "issue",
            "list",
            "--limit",
            str(_LIST_LIMIT),
            "--json",
            ISSUE_LIST_FIELDS,
        ]

        if state:
            args.extend(["--state", state])
        else:
            args.extend(["--state", "open"])

        if labels:
            for label in labels:
                args.extend(["--label", label])

        label_str = ", ".join(labels) if labels else "all"
        return self._list_json(
            args,
            limit=_LIST_LIMIT,
            kind=f"issues (labels={label_str}, state={state or 'open'})",
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

    def actions_job(self, job_id: int) -> dict[str, Any] | None:
        """Fetch a single GitHub Actions job by ID.

        Returns job data including steps[]. Used to detect infrastructure failures
        via step counts. Returns None on failure (allow_failure=True).
        """
        result = self.run(
            [
                "api",
                f"repos/{{owner}}/{{repo}}/actions/jobs/{job_id}",
            ],
            json_output=True,
            allow_failure=True,
        )
        if isinstance(result, dict):
            return result
        return None

    def check_run_annotations(self, check_run_id: int) -> list[dict[str, Any]]:
        """Fetch annotations for a specific check run.

        Returns a flat list of annotation objects. Used to detect infrastructure
        failures via billing/runner messages. Returns empty list on failure.
        """
        result = self.run(
            [
                "api",
                f"repos/{{owner}}/{{repo}}/check-runs/{check_run_id}/annotations",
            ],
            json_output=True,
            allow_failure=True,
        )
        if isinstance(result, list):
            return result
        return []

    def add_issue_label(self, number: int, label: str) -> bool:
        return self._run_bool(["issue", "edit", str(number), "--add-label", label])

    def remove_issue_label(self, number: int, label: str) -> bool:
        return self._run_bool(["issue", "edit", str(number), "--remove-label", label])

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

    def merge_pr(
        self, number: int, strategy: str, admin: bool = False, merge_flags: tuple[str, ...] = ()
    ) -> str:
        args = ["pr", "merge", str(number)]
        # merge_flags takes precedence over the legacy admin field
        if merge_flags:
            args.extend(merge_flags)
        elif admin:
            args.append(_ADMIN_FLAG)
        # Strategy flags are managed here — see ORCHESTRATOR_MANAGED_MERGE_FLAGS
        args.append(_STRATEGY_FLAGS[strategy])
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

        Note: --delete-branch is in ORCHESTRATOR_MANAGED_MERGE_FLAGS because
        it's deliberately excluded from merge_pr to avoid worktree failures.
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
                if str(issue.get("state") or "").upper() == "OPEN":
                    open_issues.add(number)
            except (GitHubError, ValueError, TypeError):
                # Issue doesn't exist or API error — treat as not blocking
                continue

        return open_issues

    def name_with_owner(self) -> str:
        """Return the repository's nameWithOwner (e.g., "owner/repo").

        Uses `gh repo view --json nameWithOwner`. Raises GitHubError on failure
        (offline, not a GitHub repo, gh missing, etc.).

        Returns:
            The repository's nameWithOwner string.
        """
        result = self.run(["repo", "view", "--json", "nameWithOwner"], json_output=True)
        if not isinstance(result, dict):
            raise GitHubError("Expected dict from gh repo view")
        name_with_owner = result.get("nameWithOwner")
        if not isinstance(name_with_owner, str):
            raise GitHubError("Expected nameWithOwner string in gh repo view output")
        return name_with_owner


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

_CLAUSE_BOUNDARY_CHARS = ".!?\n"
_ISSUE_REF = re.compile(r"#\d+")


def _clause_preceding(text: str, match_start: int) -> str:
    """Return the text of the sentence/line leading up to a match.

    Bounded by the closest preceding sentence terminator (".", "!", "?") or
    line break, so each bullet/sentence is judged independently.
    """
    boundary = max(text.rfind(ch, 0, match_start) for ch in _CLAUSE_BOUNDARY_CHARS)
    return text[boundary + 1 : match_start]


def is_infrastructure_failure(job: dict[str, Any], annotations: list[dict[str, Any]]) -> bool:
    """Detect if a failed job indicates infrastructure failure vs code failure.

    Returns True if the failed job shows signs of infrastructure failure:
    - Zero executed steps (billing lapse, runner never started)
    - Annotations matching "was not started" patterns (billing/runner issues)

    This is used to reclassify FAILURE-state checks as infra_failed instead of
    code failures, preventing rework worker dispatch against untested code.

    Args:
        job: A single job object with steps[] from the GitHub Actions API
        annotations: A flat list of annotation objects from the check-runs API

    Returns:
        True if any infrastructure failure signal is detected, False otherwise.
    """
    conclusion = str(job.get("conclusion") or "").upper()
    if conclusion != "FAILURE":
        # Only check jobs that actually failed
        return False

    steps = job.get("steps", [])
    if not isinstance(steps, list):
        steps = []

    # Signal 1: zero-step job (never started)
    # Primary signal: job with no steps at all (runner never started)
    if len(steps) == 0:
        return True

    # Fallback: filter out setup steps to detect jobs that completed
    # without running any actual test/work steps
    non_setup_steps = [
        s
        for s in steps
        if isinstance(s, dict)
        and str(s.get("name") or "").lower()
        not in {
            "set up job",
            "checkout",
            "initialize",
            "complete job",
        }
    ]

    if len(non_setup_steps) == 0:
        # Job completed with zero non-setup steps - infrastructure failure
        return True

    # Signal 2: check for "was not started" annotations
    # Billing lapse annotation: "The job was not started because recent account payments have failed or your spending limit needs to be increased."
    if not isinstance(annotations, list):
        annotations = []

    for annotation in annotations:
        if isinstance(annotation, dict):
            message = str(annotation.get("message") or "").lower()
            if "was not started" in message:
                return True

    return False


def parse_blockers(text: str) -> list[int]:
    """Parse blocker issue numbers from issue body text.

    Returns a list of issue numbers declared as blockers using patterns like:
    - "Blocked by #N"
    - "Depends on #N"
    - "Blocked-by: #N"

    Handles comma-separated lists (e.g., "Blocked by #743, #744").

    A match is only treated as the CURRENT issue declaring its own blocker
    if no other issue reference appears earlier in the same sentence/line.
    This excludes prose like "#168, #169, and #170 all build on this and are
    blocked by #159" — which describes those OTHER issues as blocked, not
    a self-declaration by whichever issue contains that text.

    Returns an empty list if no blockers are found.
    """
    if not text:
        return []

    blockers: set[int] = set()
    # Check if they appear in blocker context
    for pattern in _BLOCKER_PATTERNS:
        for match in pattern.finditer(text):
            if _ISSUE_REF.search(_clause_preceding(text, match.start())):
                continue

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
