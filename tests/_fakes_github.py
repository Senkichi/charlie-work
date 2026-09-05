"""Shared FakeGitHub fake family for the orchestrator test suite.

A single in-memory fake standing in for ``charlie_work.github.GitHub`` across
the whole suite, plus a small inheritance ladder of narrower variants that
override one or two methods (configurable checks, check-run annotations,
missing required checks, and missing-required-plus-workflow-runs). Hoisted
out of ``test_charlie_work.py`` (issue #1284) because it is imported by
every other test module that needs a fake GitHub client -- keeping one
definition here is what keeps ``test_doctor.py``'s field-constant guard
(issue #64) meaningful: there is exactly one ``FakeGitHub`` for the
orchestrator suite, not a per-file copy that can drift out of payload sync.

``test_reconcile.py`` defines its own, unrelated ``FakeGitHub`` (a
minimal reconcile-pass double, not this fake) -- see
``tests/_reconcile_fixtures.py``. The two are not related and must never be
merged or made to subclass one another.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from charlie_work import github as github_module
from charlie_work.github import _job_id_from_link, _run_id_from_link
from charlie_work.issue_linking import linked_issue_number


class FakeGitHub:
    def __init__(self, repo_root: Any = None, dry_run: bool = False) -> None:
        self.repo_root = repo_root
        self.dry_run = dry_run
        self.issues = [
            {
                "number": 123,
                "title": "Fix search",
                "url": "https://example.test/issues/123",
                "body": "Search is broken",
                "labels": [{"name": "automated-ready"}],
                "state": "OPEN",
            }
        ]
        # A janitor-green PR: open, non-draft, linked issue, tests mentioned.
        self.prs = [
            {
                "number": 456,
                "title": "Fix #123: search",
                "url": "https://example.test/pull/456",
                "headRefName": "agent/issue-123-fix-search",
                "baseRefName": "main",
                "headRefOid": "sha-abc123",
                "mergeStateStatus": "CLEAN",
                "body": "Closes #123\n\nTests: regression coverage added.",
                "labels": [],
                "isCrossRepository": False,
                "state": "OPEN",
            }
        ]
        self.labels_added: list[tuple[int, str]] = []
        self.labels_removed: list[tuple[int, str]] = []
        self.labels_created: list[tuple[str, str, str]] = []
        self.pr_labels_added: list[tuple[int, str]] = []
        self.add_pr_label_ok = True
        self.prs_created: list[dict[str, Any]] = []
        self.pr_create_return: int | None = None
        self.merged: list[tuple[int, str]] = []
        self.merged_admin_flags: list[bool] = []
        self.merged_merge_flags: list[tuple[str, ...]] = []
        self.deleted_branches: list[str] = []
        self.delete_branch_ok = True
        self.update_branch_ok = True
        self.pr_update_branch_calls: list[int] = []
        self.pr_ready_calls: list[int] = []
        self.pr_ready_ok = True
        self.pr_ready_error: str | None = None
        self.pr_close_calls: list[int] = []
        self.pr_close_ok = True
        self.pr_close_error: str | None = None
        self.pr_reopen_calls: list[int] = []
        self.pr_reopen_ok = True
        self.pr_reopen_error: str | None = None
        self.push_empty_commit_calls: list[str] = []
        self.push_empty_commit_ok = True
        self.push_empty_commit_error: str | None = None
        self.pr_head_shas: dict[int, str] = {}
        self.diffs: dict[int, str] = {}
        self.pr_external_issue_comments: dict[int, list[dict[str, Any]]] = {}
        self.pr_external_reviews: dict[int, list[dict[str, Any]]] = {}
        self.pr_external_review_comments: dict[int, list[dict[str, Any]]] = {}
        self.closed_issues: list[int] = []
        self.commits: dict[str, dict[str, Any]] = {}
        # Default base head and per-(base,head) compare overrides for testing
        # the merge-base freshness gate.
        self.base_head_sha = "base-sha"
        self.compare_overrides: dict[tuple[str, str], dict[str, Any] | None] = {}
        self.compare_diff_overrides: dict[tuple[str, str], str | None] = {}
        # Per-base branch protection overrides for testing issue #812's
        # freshness-policy derivation. Default (no override) is None, matching
        # the real GitHub.branch_protection()'s fail-closed return on any read
        # failure -- so every pre-existing test keeps exercising the
        # require_current_base fallback unchanged unless it opts in.
        self.branch_protection_overrides: dict[str, dict[str, Any] | None] = {}
        self.branch_protection_calls: list[str] = []
        self._record_pr_heads(self.prs)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name == "base_head_sha" and hasattr(self, "commits"):
            if value not in self.commits:
                self.commits[value] = {"parents": []}
        elif name == "prs" and hasattr(self, "commits") and hasattr(self, "base_head_sha"):
            self._record_pr_heads(value)

    def _record_pr_heads(self, prs: list[dict[str, Any]]) -> None:
        """Index PR head SHAs as commits rooted at the current base tip."""
        base = self.base_head_sha
        for pr in prs:
            head = pr.get("headRefOid")
            if head and head not in self.commits:
                self.commits[head] = {"parents": [{"sha": base}]}

    def check_graphql_rate_limit(self, threshold: int) -> tuple[bool, int, int | None]:
        return (True, 10000, 0)

    def invalidate_list_cache(self) -> None:
        # The real GitHub clears its per-pass list cache here (called at the
        # start of every loop pass); the fake has no cache, so this is a no-op.
        self.list_cache_invalidations = getattr(self, "list_cache_invalidations", 0) + 1

    def issue_list(self, labels=None, state=None):
        # Honor the label filter: return only issues with the ready label
        # Support both old signature (ready_label: str) and new (labels=None, state=None)
        if isinstance(labels, str):
            ready_label = labels
            issues = [
                issue
                for issue in self.issues
                if ready_label in [label["name"] for label in issue.get("labels", [])]
            ]
        elif labels:
            issues = [
                issue
                for issue in self.issues
                if any(
                    label in [label_obj["name"] for label_obj in issue.get("labels", [])]
                    for label in labels
                )
            ]
        else:
            issues = self.issues
        # Issue #1229: honor the ``state`` parameter when it is explicitly
        # passed, matching the real ``GitHub.issue_list`` (which the
        # branch-issue validator calls with ``state="open"``). When ``state``
        # is None, preserve the pre-#1229 behavior of returning all
        # label-matched issues — the real client defaults None to "open", but
        # many existing tests seed ``self.issues`` with CLOSED issues and call
        # ``issue_list()`` (no state) expecting them back, so narrowing the
        # default would break them. Callers that need open-only filtering
        # pass ``state="open"`` explicitly (as the validator does).
        if state is not None:
            wanted = state.upper()
            if wanted == "ALL":
                return issues
            issues = [
                issue for issue in issues if (issue.get("state") or "OPEN").upper() == wanted
            ]
        return issues

    def issue_view(self, number: int):
        # Return the issue matching the requested number
        for issue in self.issues:
            if issue["number"] == number:
                return issue
        raise ValueError(f"Issue {number} not found")

    def pr_list(self):
        return [pr for pr in self.prs if pr.get("state", "OPEN").upper() == "OPEN"]

    def merged_pr_list(self):
        return [pr for pr in self.prs if pr.get("state", "OPEN").upper() == "MERGED"]

    def merged_prs_for_issue(self, issue_number: int, branch_prefix: str):
        # Issue #882: match the production shape. GitHubCLI.merged_prs_for_issue
        # always returns a MergedPRSearchResult carrying ``.ok``; the base fake
        # used to return a plain list, which only worked because the sole
        # consumer reads defensively via ``getattr(merged_prs, "ok", True)``.
        # Returning the typed wrapper here keeps the fake and the real thing
        # agreeing, so a future caller that reads ``.ok`` directly does not pass
        # tests here and AttributeError in production.
        matched = []
        for pr in self.prs:
            if pr.get("state", "OPEN").upper() != "MERGED":
                continue
            bound = linked_issue_number(
                pr,
                is_cross_repository=pr.get("isCrossRepository"),
                branch_prefix=branch_prefix,
            )
            if bound == issue_number:
                matched.append(pr)
        return github_module._MergedPRSearchResult(matched, ok=True)

    def pr_view(self, number: int):
        # Return the PR matching the requested number
        for pr in self.prs:
            if pr["number"] == number:
                # Return a copy with the current head SHA (if overridden)
                pr_copy = dict(pr)
                if number in self.pr_head_shas:
                    pr_copy["headRefOid"] = self.pr_head_shas[number]
                return pr_copy
        raise ValueError(f"PR {number} not found")

    def pr_create(self, head: str, base: str, title: str, body: str) -> int | None:
        self.prs_created.append({"head": head, "base": base, "title": title, "body": body})
        return self.pr_create_return

    def pr_commits(self, number: int) -> list[dict[str, Any]] | None:
        # No fixture data configured means an empty list, matching the real
        # GitHub.pr_commits's "no failure, nothing found" shape rather than
        # raising. Not exercised by any GitHubLike-typed call site as of the
        # PR that added this method (only the concrete GitHub-typed
        # closing-keyword-check CLI path calls it), but kept here so
        # FakeGitHub stays a complete stand-in for the GitHubLike protocol.
        return []

    def pr_checks(self, number: int):
        return [
            {"name": "Tests passed", "state": "SUCCESS"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]

    def check_run_annotations(self, check_run_id: int) -> list[dict[str, Any]]:
        # Mirrors the real GitHub.check_run_annotations default: no
        # annotations configured means an empty list, never a raise.
        return []

    def pr_diff(self, number: int):
        # Return custom diff if set, otherwise default
        if number in self.diffs:
            return self.diffs[number]
        return "diff --git a/file b/file"

    def add_issue_label(self, number: int, label: str) -> bool:
        self.labels_added.append((number, label))
        return True

    def remove_issue_label(self, number: int, label: str) -> bool:
        self.labels_removed.append((number, label))
        return True

    def add_pr_label(self, number: int, label: str) -> bool:
        self.pr_labels_added.append((number, label))
        return self.add_pr_label_ok

    def close_issue(self, number: int) -> bool:
        """Track issue closure for testing. Idempotent — returns True even if already closed."""
        # Track the closure
        self.closed_issues.append(number)
        # Update the issue state in the issues list
        for issue in self.issues:
            if issue["number"] == number:
                issue["state"] = "CLOSED"
                break
        return True

    def issue_comment(self, number: int, body_file: Path) -> None:
        """Record issue comments posted by the orchestrator (issue #1000)."""
        posted = getattr(self, "issue_comments_posted", [])
        posted.append((number, Path(body_file).read_text(encoding="utf-8")))
        self.issue_comments_posted = posted

    def name_with_owner(self) -> str:
        return "test-owner/test-repo"

    def merge_pr(
        self, number: int, strategy: str, admin: bool = False, merge_flags: tuple[str, ...] = ()
    ) -> str:
        self.merged.append((number, strategy))
        # merge_flags takes precedence over admin
        if merge_flags:
            self.merged_admin_flags.append("--admin" in merge_flags)
        else:
            self.merged_admin_flags.append(admin)
        self.merged_merge_flags.append(merge_flags)

        # Model the real effect of a merge: the base branch tip advances to a
        # merge commit whose parents are the previous base tip and the merged PR
        # head. This lets stale-base tests derive base movement organically from
        # recorded merges instead of hand-feeding compare_overrides.
        pr: dict[str, Any] | None = None
        for candidate in self.prs:
            if candidate.get("number") == number:
                pr = candidate
                break
        if pr is not None:
            base_ref = pr.get("baseRefName") or "main"
            head_sha = pr.get("headRefOid")
            old_base = self.base_head_sha
            merge_sha = f"{base_ref}-merged-{head_sha}"
            self.commits[merge_sha] = {
                "parents": [{"sha": old_base}, {"sha": head_sha}],
                "committer": {"login": "web-flow"},
                "commit": {"committer": {"name": "GitHub"}},
            }
            self.base_head_sha = merge_sha

        return "merged"

    def delete_branch(self, branch: str) -> bool:
        self.deleted_branches.append(branch)
        return self.delete_branch_ok

    def pr_ready(self, number: int) -> github_module.GitHubRunResult:
        self.pr_ready_calls.append(number)
        if self.pr_ready_ok:
            for pr in self.prs:
                if pr["number"] == number:
                    # Simulate GitHub's real effect: the PR is no longer a
                    # draft, so the next janitor pass sees isDraft=False.
                    pr["isDraft"] = False
                    break
            return github_module.GitHubRunResult(
                ok=True, returncode=0, stdout="", stderr="", value=None, error=None
            )
        error = self.pr_ready_error or "gh: pull request #%d is not ready for review" % number
        return github_module.GitHubRunResult(
            ok=False, returncode=1, stdout="", stderr=error, value=None, error=error
        )

    def pr_close(self, number: int) -> github_module.GitHubRunResult:
        self.pr_close_calls.append(number)
        if self.pr_close_ok:
            for pr in self.prs:
                if pr["number"] == number:
                    pr["state"] = "CLOSED"
                    break
            return github_module.GitHubRunResult(
                ok=True, returncode=0, stdout="", stderr="", value=None, error=None
            )
        error = self.pr_close_error or "gh: could not close pull request #%d" % number
        return github_module.GitHubRunResult(
            ok=False, returncode=1, stdout="", stderr=error, value=None, error=error
        )

    def pr_reopen(self, number: int) -> github_module.GitHubRunResult:
        self.pr_reopen_calls.append(number)
        if self.pr_reopen_ok:
            for pr in self.prs:
                if pr["number"] == number:
                    pr["state"] = "OPEN"
                    break
            return github_module.GitHubRunResult(
                ok=True, returncode=0, stdout="", stderr="", value=None, error=None
            )
        error = self.pr_reopen_error or "gh: could not reopen pull request #%d" % number
        return github_module.GitHubRunResult(
            ok=False, returncode=1, stdout="", stderr=error, value=None, error=error
        )

    def push_empty_commit(self, branch: str) -> github_module.GitHubRunResult:
        self.push_empty_commit_calls.append(branch)
        if self.push_empty_commit_ok:
            for pr in self.prs:
                if pr.get("headRefName") == branch:
                    # Simulate GitHub's real effect: the branch tip moves to a
                    # new (synthetic) SHA, same as a real empty-commit push.
                    old_head = pr.get("headRefOid", "")
                    pr["headRefOid"] = f"{old_head}-empty-commit"
                    break
            return github_module.GitHubRunResult(
                ok=True, returncode=0, stdout="", stderr="", value=None, error=None
            )
        error = self.push_empty_commit_error or f"gh: could not push empty commit to {branch!r}"
        return github_module.GitHubRunResult(
            ok=False, returncode=1, stdout="", stderr=error, value=None, error=error
        )

    def pr_update_branch(self, pr_number: int) -> bool:
        self.pr_update_branch_calls.append(pr_number)
        # Simulate a base update by moving the PR's head to a new SHA
        # This reproduces the churn that the fix prevents
        for pr in self.prs:
            if pr["number"] == pr_number:
                # Append a merge-SHA marker to simulate the head moving
                old_head = pr.get("headRefOid", "")
                new_head = f"{old_head}-updated"
                pr["headRefOid"] = new_head
                # A real update-branch makes the PR current with its base. Future
                # compare calls for the new head should see the current base tip.
                pr["mergeStateStatus"] = "CLEAN"
                base_ref = pr.get("baseRefName") or "main"
                base_head = self.base_head_sha
                self.compare_overrides[(base_ref, new_head)] = {
                    "base_commit": {"sha": base_head},
                    "merge_base_commit": {"sha": base_head},
                }
                # Record the fake commit metadata so the post-sync verification
                # helper sees a valid GitHub web-flow merge commit.
                self.commits[new_head] = {
                    "parents": [
                        {"sha": old_head},
                        {"sha": base_head},
                    ],
                    "committer": {"login": "web-flow"},
                    "commit": {"committer": {"name": "GitHub"}},
                }
                return self.update_branch_ok
        return False

    def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
        """Default implementation: check the actual state field in issues."""
        open_issues: set[int] = set()
        for number in issue_numbers:
            for issue in self.issues:
                if issue["number"] == number and str(issue.get("state") or "").upper() == "OPEN":
                    open_issues.add(number)
                    break
        return open_issues

    def run(self, args: list[str], *, json_output: bool = False, allow_failure: bool = False):
        """Fake run method for GitHub API calls. Returns empty list for dependencies by default."""
        # Handle dependency API calls
        if "dependencies" in " ".join(args):
            # Default: return empty list (feature not available)
            # Tests can override this by setting dependencies_response
            if hasattr(self, "dependencies_response"):
                return self.dependencies_response
            return [] if json_output else ""
        # Handle run list API calls
        if "run" in args and "list" in args:
            # Default: return empty list
            # Tests can override this by setting runs_response
            if hasattr(self, "runs_response"):
                return self.runs_response
            return [] if json_output else ""
        # Handle run cancel API calls
        if "run" in args and "cancel" in args:
            # Default: return success string
            return "Cancelled"
        # Handle external findings API calls (issue #950).
        joined = " ".join(args)
        m = re.search(r"/issues/(\d+)/comments", joined)
        if m and "/pulls/" not in joined:
            return self.pr_external_issue_comments.get(int(m.group(1)), [])
        m = re.search(r"/pulls/(\d+)/reviews", joined)
        if m and "/comments" not in joined:
            return self.pr_external_reviews.get(int(m.group(1)), [])
        m = re.search(r"/pulls/(\d+)/comments", joined)
        if m and "/reviews/" not in joined:
            return self.pr_external_review_comments.get(int(m.group(1)), [])
        # Handle paginated PR list REST API calls from reconcile.py.
        if args[0] == "api" and "pulls?state=all" in args[1]:
            url = args[1]
            page_match = re.search(r"[?&]page=(\d+)", url)
            page = int(page_match.group(1)) if page_match else 1
            per_page_match = re.search(r"[?&]per_page=(\d+)", url)
            per_page = int(per_page_match.group(1)) if per_page_match else 100
            start = (page - 1) * per_page
            return self.prs[start : start + per_page]
        # Handle paginated issue list REST API calls from reconcile.py.
        if args[0] == "api" and "issues?state=all" in args[1]:
            url = args[1]
            page_match = re.search(r"[?&]page=(\d+)", url)
            page = int(page_match.group(1)) if page_match else 1
            per_page_match = re.search(r"[?&]per_page=(\d+)", url)
            per_page = int(per_page_match.group(1)) if per_page_match else 100
            start = (page - 1) * per_page
            return self.issues[start : start + per_page]
        # Handle other API calls (for reconcile tests)
        if json_output:
            return []
        return ""

    def commit(self, sha: str) -> github_module.GitHubRunResult:
        commit = self.commits.get(sha)
        if not isinstance(commit, dict):
            return github_module.GitHubRunResult(
                ok=False,
                returncode=1,
                stdout="",
                stderr="",
                value=None,
                error=f"commit {sha} not found",
            )
        return github_module.GitHubRunResult(
            ok=True, returncode=0, stdout="", stderr="", value=commit
        )

    def _ancestors(self, sha: str) -> set[str]:
        """Return all ancestors of ``sha`` (including ``sha`` itself)."""
        seen: set[str] = set()
        stack = [sha]
        while stack:
            current = stack.pop()
            if current in seen or not current:
                continue
            seen.add(current)
            commit = self.commits.get(current)
            if not isinstance(commit, dict):
                continue
            for parent in commit.get("parents", []):
                if isinstance(parent, dict):
                    parent_sha = parent.get("sha")
                else:
                    parent_sha = parent
                if parent_sha:
                    stack.append(parent_sha)
        return seen

    def _merge_base(self, base_sha: str, head_sha: str) -> str | None:
        """Return the best common ancestor of ``base_sha`` and ``head_sha``.

        The best common ancestor is a common ancestor that is not itself an
        ancestor of another common ancestor. For linear DAGs this is the usual
        merge-base; the simple filter works for the small graphs in these tests.
        """
        base_ancestors = self._ancestors(base_sha)
        head_ancestors = self._ancestors(head_sha)
        common = base_ancestors & head_ancestors
        if not common:
            return None
        best = [
            sha
            for sha in common
            if not any(sha in self._ancestors(other) and sha != other for other in common)
        ]
        if not best:
            best = list(common)

        # Deterministic tie-break: prefer the ancestor closest to the base tip.
        def _distance(source: str, target: str) -> int:
            if source == target:
                return 0
            visited: set[str] = {source}
            queue: list[tuple[str, int]] = [(source, 0)]
            while queue:
                current, dist = queue.pop(0)
                commit = self.commits.get(current)
                if not isinstance(commit, dict):
                    continue
                for parent in commit.get("parents", []):
                    if isinstance(parent, dict):
                        parent_sha = parent.get("sha")
                    else:
                        parent_sha = parent
                    if parent_sha == target:
                        return dist + 1
                    if parent_sha and parent_sha not in visited:
                        visited.add(parent_sha)
                        queue.append((parent_sha, dist + 1))
            return len(self.commits)

        best.sort(key=lambda sha: (_distance(base_sha, sha), _distance(head_sha, sha), sha))
        return best[0]

    def compare(self, base: str, head: str) -> dict[str, Any] | None:
        override = self.compare_overrides.get((base, head))
        if override is not None:
            return override
        base_head = self.base_head_sha

        # Find the matching PR so we can honor mergeStateStatus hints when the
        # graph is not enough or contradicts a BEHIND signal.
        matching_pr = None
        for pr in self.prs:
            if pr.get("headRefOid") == head:
                matching_pr = pr
                break
        if matching_pr is None:
            for pr_number, pr_head in self.pr_head_shas.items():
                if pr_head == head:
                    for pr in self.prs:
                        if pr.get("number") == pr_number:
                            matching_pr = pr
                            break
                    break

        # If we have a commit graph for both the current base tip and the head,
        # derive the merge base from recorded merges. This is the path that lets
        # merge tests prove ``merge advances main`` organically.
        if base_head in self.commits and head in self.commits:
            merge_base = self._merge_base(base_head, head)
            base_current = merge_base == base_head
            if base_current and str(matching_pr.get("mergeStateStatus") or "").upper() == "BEHIND":
                # A BEHIND mergeStateStatus is a stronger stale signal than the
                # current graph, so tests can still simulate a stale branch by
                # setting mergeStateStatus to BEHIND.
                return {
                    "base_commit": {"sha": base_head},
                    "merge_base_commit": {"sha": f"{base_head}-stale"},
                }
            return {
                "base_commit": {"sha": base_head},
                "merge_base_commit": {"sha": merge_base if merge_base else ""},
            }

        # If no graph is available, fall back to the PR's mergeStateStatus when
        # it is known. Tests can still use compare_overrides to model exceptional
        # cases (e.g. CLEAN-but-stale where mergeStateStatus lags).
        if (
            matching_pr is not None
            and str(matching_pr.get("mergeStateStatus") or "").upper() == "BEHIND"
        ):
            return {
                "base_commit": {"sha": base_head},
                "merge_base_commit": {"sha": f"{base_head}-stale"},
            }
        # Default: the PR's merge-base is the current base tip.
        return {
            "base_commit": {"sha": base_head},
            "merge_base_commit": {"sha": base_head},
        }

    def compare_diff(self, base: str, head: str) -> str | None:
        override = self.compare_diff_overrides.get((base, head), "_unset")
        if override != "_unset":
            return override
        return f"diff --git a/interdiff b/interdiff\n--- a/interdiff\n+++ b/interdiff\n@@ -1 +1 @@\n-{base}\n+{head}\n"

    def branch_protection(self, base: str) -> dict[str, Any] | None:
        self.branch_protection_calls.append(base)
        return self.branch_protection_overrides.get(base)

    def label_create(self, label: str, color: str, description: str) -> None:
        self.labels_created.append((label, color, description))

    def label_list(self) -> list[dict[str, object]]:
        # Return all labels that have been created — simulates creation success.
        return [{"name": name} for name, _color, _desc in self.labels_created]

    def pr_comment(self, number: int, body_file: Path) -> None:
        pass

    def remove_pr_label(self, number: int, label: str) -> bool:
        return True

    def actions_job(self, job_id: int) -> dict[str, Any] | None:
        return None

    def commit_check_runs(self, sha: str) -> list[dict[str, Any]] | None:
        return None

    def workflow_runs_for_head(self, head_sha: str) -> list[dict[str, Any]] | None:
        return None

    def validate_field_lists(self) -> None:
        pass


class FakeGitHubWithChecks(FakeGitHub):
    """FakeGitHub whose pr_checks returns a configurable list."""

    def __init__(self, checks: list[dict[str, Any]] | None = None) -> None:
        super().__init__()
        self.checks = checks if checks is not None else []

    def pr_checks(self, number: int) -> list[dict[str, Any]]:
        # Mirror production GitHub.pr_checks: inject databaseId/runId from the
        # check link, but only when not already provided by the test.
        return [
            {
                **check,
                "databaseId": check.get("databaseId", _job_id_from_link(check.get("link"))),
                "runId": check.get("runId", _run_id_from_link(check.get("link"))),
            }
            for check in self.checks
        ]


class FakeGitHubWithChecksAndAnnotations(FakeGitHubWithChecks):
    """FakeGitHubWithChecks whose check_run_annotations returns a configurable
    per-check-run-id mapping (issue #771 tests)."""

    def __init__(
        self,
        checks: list[dict[str, Any]] | None = None,
        annotations_by_check_run_id: dict[int, list[dict[str, Any]]] | None = None,
    ) -> None:
        super().__init__(checks=checks)
        self.annotations_by_check_run_id = annotations_by_check_run_id or {}

    def check_run_annotations(self, check_run_id: int) -> list[dict[str, Any]]:
        return self.annotations_by_check_run_id.get(check_run_id, [])


class FakeGitHubWithMissingRequired(FakeGitHubWithChecks):
    """No required checks present at all, so every required check is missing."""

    def __init__(self) -> None:
        super().__init__(checks=[])


class FakeGitHubWithMissingRequiredAndRuns(FakeGitHubWithMissingRequired):
    """No required checks reported at all (so every required check is
    "missing" per the janitor gate), plus a configurable
    ``workflow_runs_for_head`` response so tests can control whether GitHub
    Actions ever created a run for the head SHA.
    """

    def __init__(self, runs: list[dict[str, Any]] | None) -> None:
        super().__init__()
        self._runs = runs

    def workflow_runs_for_head(self, head_sha: str) -> list[dict[str, Any]] | None:
        return self._runs
