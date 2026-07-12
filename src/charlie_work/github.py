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
# NOTE: "databaseId" is NOT a valid `gh pr checks --json` field (unlike `gh run
# list --json`, which does support it) — installed gh CLIs reject it with
# 'Unknown JSON field: "databaseId"' and exit non-zero. Because pr_checks() calls
# run(..., allow_failure=True) and treats a non-list result as "no checks", adding
# it here silently returns [] from every pr_checks() call, which makes
# summarize_checks() report all required checks "missing" and merge_ready()
# compute can_merge=False for every PR — the entire auto-merge lane goes
# silently dead (regression introduced 2026-07-10, fixed same day). The GitHub
# Actions job id (needed for infrastructure-failure classification, issue #210)
# is instead derived from "link" by pr_checks() via _job_id_from_link() and
# injected back into each check dict as "databaseId", so downstream consumers
# (workflow.py) see an unchanged contract.
PR_CHECKS_FIELDS = "name,state,bucket,link"
LABEL_LIST_FIELDS = "name"
# Minimal field lists for drift detection (reconcile.py)
RECONCILE_PR_FIELDS = (
    "number,title,url,headRefName,baseRefName,body,state,labels,isCrossRepository"
)
RECONCILE_ISSUE_FIELDS = "number,title,url,body,labels"
RUN_LIST_FIELDS = "databaseId,status,createdAt,headBranch"

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
class GitHubRunResult:
    """Result of a ``gh`` invocation when ``allow_failure=True``.

    Errors stay as values: callers check ``ok`` and ``error`` and only use
    ``value`` when ``ok`` is True. ``value`` is the parsed JSON (when
    ``json_output=True``) or the captured stdout (when ``json_output=False``).
    """

    ok: bool
    returncode: int
    stdout: str
    stderr: str
    value: Any | None = None
    error: str | None = None


# Matches the job-id segment of a GitHub Actions check link, e.g.
# https://github.com/OWNER/REPO/actions/runs/RUN_ID/job/JOB_ID (optionally
# followed by a query string or #fragment, e.g. "?check_suite_focus=true").
_ACTIONS_JOB_LINK_RE = re.compile(r"/actions/runs/\d+/job/(\d+)")


def _job_id_from_link(link: str | None) -> int | None:
    """Derive a GitHub Actions job id from a check's ``link`` field.

    ``gh pr checks --json`` has no ``databaseId`` field, so the job id (needed
    to call ``actions_job``/``check_run_annotations`` for infrastructure-failure
    classification, issue #210) must be parsed out of the check's link. Only
    GitHub Actions check links match; external status checks may have
    arbitrary or empty links. Never raises — returns None for anything that
    doesn't match.
    """
    if not link:
        return None
    match = _ACTIONS_JOB_LINK_RE.search(link)
    if not match:
        return None
    return int(match.group(1))


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
                check=False,
            )
        except FileNotFoundError as exc:
            if allow_failure:
                return GitHubRunResult(
                    ok=False,
                    returncode=0,
                    stdout="",
                    stderr="",
                    value=None,
                    error="GitHub CLI `gh` is not installed or not on PATH.",
                )
            raise GitHubError("GitHub CLI `gh` is not installed or not on PATH.") from exc

        output = result.stdout.strip()

        if not allow_failure:
            if result.returncode != 0:
                raise GitHubError(
                    result.stderr.strip() or result.stdout.strip() or str(result.returncode)
                )
            if not json_output:
                return output
            if not output:
                return None
            try:
                return json.loads(output)
            except json.JSONDecodeError as exc:
                raise GitHubError(f"Expected JSON from gh command: {' '.join(command)}") from exc

        # allow_failure=True: always return a structured result so callers can
        # distinguish command failure from empty-but-legitimate output.
        error: str | None = None
        value: Any | None = None
        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip() or f"gh exited {result.returncode}"
        if not json_output:
            value = output if result.returncode == 0 else None
        elif output:
            try:
                value = json.loads(output)
            except json.JSONDecodeError as exc:
                error = f"Expected JSON from gh command: {' '.join(command)}"
                value = None
        return GitHubRunResult(
            ok=result.returncode == 0 and error is None,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            value=value,
            error=error,
        )

    def pr_create(
        self,
        head: str,
        base: str,
        title: str,
        body: str,
    ) -> int | None:
        """Create a GitHub PR for ``head`` into ``base``.

        Returns the PR number, or ``None`` if creation failed or the local ``gh``
        does not support JSON output. Errors are returned as values, never raised.
        """
        if self.dry_run:
            return 0
        try:
            result = self.run(
                [
                    "pr",
                    "create",
                    "--head",
                    head,
                    "--base",
                    base,
                    "--title",
                    title,
                    "--body",
                    body,
                    "--json",
                    "number",
                ],
                json_output=True,
            )
        except GitHubError:
            return None
        if isinstance(result, dict) and "number" in result:
            return int(result["number"])
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict) and "number" in first:
                return int(first["number"])
        return None

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
        if isinstance(result, GitHubRunResult):
            return result.value if result.ok else ""
        return result if isinstance(result, str) else ""

    def pr_checks(self, number: int) -> list[dict[str, Any]] | None:
        result = self.run(
            ["pr", "checks", str(number), "--json", PR_CHECKS_FIELDS],
            json_output=True,
            allow_failure=True,
        )
        if isinstance(result, GitHubRunResult):
            # gh pr checks exits non-zero both when the command itself fails (e.g.
            # an unsupported JSON field) and when checks are failing. The
            # difference is in the value: a command failure yields no parseable
            # list, while genuinely failing checks still produce a list of results.
            if isinstance(result.value, list):
                checks = result.value
            elif result.ok and result.value is None:
                # Empty successful response (no checks reported) is legitimate.
                return []
            else:
                # Command-level failure (Unknown JSON field, GraphQL error, etc.)
                return None
        else:
            # Legacy pre-result-object fallback
            checks = result if isinstance(result, list) else []
        # gh pr checks --json has no databaseId field; derive the GitHub Actions
        # job id from "link" and inject it so downstream consumers (workflow.py)
        # keep reading check.get("databaseId") unchanged.
        return [{**check, "databaseId": _job_id_from_link(check.get("link"))} for check in checks]

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
        if isinstance(result, GitHubRunResult):
            return result.value if result.ok and isinstance(result.value, dict) else None
        return result if isinstance(result, dict) else None

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
        if isinstance(result, GitHubRunResult):
            return result.value if result.ok and isinstance(result.value, list) else []
        return result if isinstance(result, list) else []

    def validate_field_lists(self) -> None:
        """Validate the compile-time ``--json`` field lists against ``gh``.

        Probes each list with an invalid field and parses the ``Available
        fields:`` section of the stderr. Fails fast with a ``ConfigError`` naming
        the constant and the offending field(s) when the installed ``gh`` CLI
        does not support a configured field.
        """
        # Import lazily to avoid the config -> github import cycle.
        from .config import ConfigError

        probe = "nonexistent"  # Invalid field name, gh will list valid ones
        field_lists: list[tuple[str, list[str], str]] = [
            ("ISSUE_LIST_FIELDS", ["issue", "list", "--state", "open", "--limit", "1", "--json", probe], ISSUE_LIST_FIELDS),
            ("ISSUE_VIEW_FIELDS", ["issue", "view", "0", "--json", probe], ISSUE_VIEW_FIELDS),
            ("PR_LIST_FIELDS", ["pr", "list", "--state", "open", "--limit", "1", "--json", probe], PR_LIST_FIELDS),
            ("PR_VIEW_FIELDS", ["pr", "view", "0", "--json", probe], PR_VIEW_FIELDS),
            ("PR_CHECKS_FIELDS", ["pr", "checks", "0", "--json", probe], PR_CHECKS_FIELDS),
            ("LABEL_LIST_FIELDS", ["label", "list", "--limit", "1", "--json", probe], LABEL_LIST_FIELDS),
            ("RECONCILE_PR_FIELDS", ["pr", "list", "--state", "all", "--limit", "1", "--json", probe], RECONCILE_PR_FIELDS),
            ("RECONCILE_ISSUE_FIELDS", ["issue", "list", "--state", "open", "--limit", "1", "--json", probe], RECONCILE_ISSUE_FIELDS),
            ("RUN_LIST_FIELDS", ["run", "list", "--limit", "1", "--json", probe], RUN_LIST_FIELDS),
        ]

        for name, args, fields in field_lists:
            try:
                result = subprocess.run(
                    ["gh", *args],
                    cwd=self.repo_root,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise ConfigError("GitHub CLI `gh` is not installed or not on PATH") from exc

            if result.returncode == 0:
                raise ConfigError(
                    f"gh did not reject invalid field for {name}; cannot validate field list"
                )

            stderr = result.stderr
            if "Unknown JSON field" not in stderr and "Available fields" not in stderr:
                raise ConfigError(
                    f"Could not validate field list {name}: {stderr.strip() or result.stdout.strip()}"
                )

            available: set[str] = set()
            match = re.search(r"Available fields:\n((?:  .+\n)+)", stderr)
            if match:
                available = {line.strip() for line in match.group(1).splitlines() if line.strip()}

            unsupported = [field for field in fields.split(",") if field not in available]
            if unsupported:
                raise ConfigError(
                    f"gh does not support field(s) for {name}: {', '.join(unsupported)}"
                )

    def add_issue_label(self, number: int, label: str) -> bool:
        return self._run_bool(["issue", "edit", str(number), "--add-label", label])

    def remove_issue_label(self, number: int, label: str) -> bool:
        return self._run_bool(["issue", "edit", str(number), "--remove-label", label])

    def close_issue(self, number: int) -> bool:
        """Close an issue. Idempotent — returns True even if already closed.

        Uses `gh issue close`. Returns True on success, False on failure.
        Never raises — per-issue failures are reported as values and must not
        abort a batch operation.
        """
        try:
            self.run(["issue", "close", str(number)])
            return True
        except GitHubError:
            return False

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


def detect_prose_only_dependencies(text: str) -> bool:
    """Detect prose-only dependency declarations in issue body.

    Returns True if the issue body contains dependency-like prose without
    structured blocker declarations. This catches cases like "Do not dispatch
    before P2-T2/P2-T3 have landed" that lack corresponding "Blocked by #N" markers.

    Patterns detected:
    - "do not dispatch before" (case-insensitive)
    - "depends on <...> P\\d+-T\\d+" — task reference in dependency context
    - "wait for <...> P\\d+-T\\d+ <...> (complete|done|land|merge|ship)" — task
      reference with completion verb; covers "Wait for P1-T5 to complete first."
    - "before/until/after <...> P\\d+-T\\d+ <...> (land|merge|complete|done|ship)"
    - "wait for" before a PR or merge event (non-task dependency prose)

    Pattern 2 is intentionally scoped to dependency context only — bare task
    marker mentions like "implements P2-T4" or title suffixes "(P2-T4)" are
    NOT matched, to avoid flagging every plan-generated issue for human review.

    Args:
        text: The issue body text to check

    Returns:
        True if prose-only dependencies are detected, False otherwise
    """
    if not text:
        return False

    # Pattern 1: "do not dispatch before" and variants
    if re.search(r"do\s+not\s+dispatch\s+before", text, flags=re.IGNORECASE):
        return True

    # Pattern 2: task references (P\d+-T\d+) only in dependency context.
    # "depends on ... P\d+-T\d+" — classic self-declaration
    if re.search(r"depends\s+on\s+[^.\n]*P\d+-T\d+", text, flags=re.IGNORECASE):
        return True
    # "wait for ... P\d+-T\d+ ... <completion verb>" — e.g. "Wait for P1-T5 to complete first."
    if re.search(
        r"wait\s+for\s+[^.\n]*P\d+-T\d+[^.\n]*(?:land|merge|complete|done|ship)",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    # "before/until/after ... P\d+-T\d+ ... <completion verb>"
    if re.search(
        r"(?:before|until|after)\s+[^.\n]*P\d+-T\d+[^.\n]*(?:land|merge|complete|done|ship)",
        text,
        flags=re.IGNORECASE,
    ):
        return True

    # Pattern 3: "wait for" before a PR or merge event (non-task dependency prose)
    if re.search(
        r"wait\s+for\s+(?:this|that|these|those)?\s*(?:PR|merge|land)", text, flags=re.IGNORECASE
    ):
        return True

    return False


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
    if isinstance(result, GitHubRunResult):
        if not result.ok:
            # Transient error or gh not available — fail open with warning
            logger.warning(
                "GitHub dependencies API failed for issue #%d: %s - treating as no dependencies",
                issue_number,
                result.error,
            )
            return []
        value = result.value
    else:
        value = result

    if value is None:
        # Legacy FakeGitHub may return None for an allow_failure=True call; in
        # production gh.run returns a GitHubRunResult with ok=False and error.
        logger.warning(
            "GitHub dependencies API returned None for issue #%d - treating as no dependencies",
            issue_number,
        )
        return []
    elif isinstance(value, dict):
        # 404/410 error response — feature not available on this repo
        # This is expected for repos without dependencies enabled
        return []
    elif isinstance(value, list):
        # Extract issue numbers from the dependency list
        return [int(dep.get("number", 0)) for dep in value if dep.get("number")]
    else:
        # Unexpected type — fail open with warning
        logger.warning(
            "GitHub dependencies API returned unexpected type %s for issue #%d - treating as no dependencies",
            type(value),
            issue_number,
        )
        return []


def cancel_superseded_runs(
    gh: GitHub,
    default_branch: str,
    workflow_name: str,
) -> dict[str, Any]:
    """Cancel superseded queued runs on the default branch for a workflow.

    Lists QUEUED runs for the given workflow on the default branch, keeps the
    newest (by createdAt, not run ID), and cancels the rest via `gh run cancel`.
    Never cancels in_progress runs; never touches PR-branch runs.

    Args:
        gh: GitHub client instance
        default_branch: The default branch name (e.g., "main")
        workflow_name: The workflow name to filter runs

    Returns:
        Dict with cancellation results:
        {
            "total_queued": int,
            "kept": int,
            "cancelled": int,
            "cancelled_run_ids": list[int],
            "errors": list[str],
        }
    """
    result = {
        "total_queued": 0,
        "kept": 0,
        "cancelled": 0,
        "cancelled_run_ids": [],
        "errors": [],
    }

    if not workflow_name:
        result["errors"].append("workflow_name is empty - cannot cancel runs")
        return result

    try:
        # List queued runs for the workflow on the default branch
        runs = gh.run(
            [
                "run",
                "list",
                "--workflow",
                workflow_name,
                "--branch",
                default_branch,
                "--status",
                "queued",
                "--limit",
                "100",
                "--json",
                RUN_LIST_FIELDS,
            ],
            json_output=True,
            allow_failure=True,
        )

        if isinstance(runs, GitHubRunResult):
            if not runs.ok or not isinstance(runs.value, list):
                result["errors"].append(
                    f"Expected list from gh run list, got {type(runs.value)} (error: {runs.error})"
                )
                return result
            runs_list = runs.value
        else:
            if not isinstance(runs, list):
                result["errors"].append(f"Expected list from gh run list, got {type(runs)}")
                return result
            runs_list = runs

        queued_runs = [r for r in runs_list if r.get("status") == "queued"]
        result["total_queued"] = len(queued_runs)

        if len(queued_runs) <= 1:
            # 0 or 1 queued runs - nothing to cancel
            result["kept"] = len(queued_runs)
            return result

        # Sort by createdAt (newest first) to keep the newest
        queued_runs.sort(key=lambda r: r.get("createdAt", ""), reverse=True)

        # Keep the newest, cancel the rest
        to_cancel = queued_runs[1:]

        result["kept"] = 1

        for run in to_cancel:
            run_id = run.get("databaseId")
            if not isinstance(run_id, int):
                result["errors"].append(f"Run missing databaseId: {run}")
                continue

            try:
                cancel_result = gh.run(["run", "cancel", str(run_id)], allow_failure=True)
                # With allow_failure=True, gh.run returns a structured result. A
                # dry-run string is also truthy. Count as cancelled only when the
                # result indicates success (or dry-run).
                if isinstance(cancel_result, GitHubRunResult):
                    cancelled = cancel_result.ok
                else:
                    cancelled = cancel_result is not None
                if cancelled:
                    result["cancelled_run_ids"].append(run_id)
                    result["cancelled"] += 1
                else:
                    result["errors"].append(f"Failed to cancel run {run_id}")
            except GitHubError as e:
                result["errors"].append(f"Failed to cancel run {run_id}: {e}")

    except GitHubError as e:
        result["errors"].append(f"GitHub API error: {e}")

    return result
