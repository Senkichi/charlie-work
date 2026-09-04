from __future__ import annotations

import json
import logging
import random
import re
import subprocess
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .config import RuntimeConfig

# LOAD-BEARING RE-EXPORT — NOT AN UNUSED IMPORT. Do not delete; the `noqa`
# below marks a deliberate re-export, not a lint concession.
#
# GitHubError was MOVED to ci_fleet, not copied, because it is *caught*. Two
# structurally identical exception classes are unrelated types, so a local
# re-declaration would stop every `except GitHubError` in charlie_work from
# catching what ci_fleet raises — with no import error and no failure at the
# raise site. Re-exported here rather than at the adapter because consumers
# already do `from .github import GitHubError`.
#
# Measured 2026-08-05 by AST, not grep — a line matcher misses parenthesized
# multi-line imports and counts non-handler mentions, and it errs in both
# directions at once, so two greps agreeing is not corroboration. An earlier
# version of this comment claimed "16 modules" and was wrong: 16 is exactly
# workflow.py's own handler count, which is where the number came from.
# Actual: 7 modules import the name and 37 `except` handlers across 6 files
# depend on it being ci_fleet's class (independently reproduced ci_fleet-side).
# Counts are indicative only — `tests/test_ci_fleet_seams.py` is the guard, and
# it asserts the identity directly, so deleting or re-declaring this fails the
# suite rather than degrading silently. Fix the seam, never the assertion.
from ci_fleet.github import GitHubError  # noqa: F401  (deliberate re-export)

from .checks import _run_id_from_link
from .github_capabilities import (
    ChecksLike,
    CommentsLike,
    IssuesLike,
    LABEL_LIST_FIELDS,
    LabelsLike,
    MergeBranchLike,
    PullRequestsLike,
    RepoMetaLike,
)
from .github_delegation import _COLLABORATORS, _install_delegates
from .github_delegation import _ROUTES, _SIGNATURE_SOURCE, _make_delegate  # noqa: F401 (deliberate re-export)
from .subprocess_runner import no_console_window_kwargs

logger = logging.getLogger(__name__)

_LIST_LIMIT = 500

# Defaults used when GitHub is constructed without a RuntimeConfig (tests and
# legacy callers). Production code should pass config.runtime so these are
# configurable via orchestrator.config.yaml.
_DEFAULT_GH_MAX_RETRIES = 3
_DEFAULT_GH_RETRY_BASE_SECONDS = 1.0
_DEFAULT_GH_TIMEOUT_SECONDS = 120.0
# Conventional exit status for "killed by timeout" (GNU coreutils `timeout`).
# A TimeoutExpired carries no returncode of its own, and callers that branch on
# returncode must not see a 0 that reads as success.
_TIMEOUT_RETURNCODE = 124
_DEFAULT_GRAPHQL_RATE_LIMIT_THRESHOLD = 1500

# Bound on concurrent `gh` subprocesses spawned by are_issues_open() for a
# single batch of cache misses. Each `gh` call is I/O-bound (process spawn +
# network round trip, ~1-7s observed), so this is a fan-out width, not a CPU
# budget -- picked to keep well clear of GitHub's secondary rate limits while
# still cutting a serial N x ~2s loop down substantially. See issue #870.
_MAX_ISSUE_STATE_WORKERS = 8

# How many issue numbers to pack into one batched `gh api graphql` query.
# Kept conservative to stay under the ~32KB Windows command-line limit and
# GitHub's GraphQL node/complexity budgets. See issue #923.
_GRAPHQL_BATCH_SIZE = 50

# GitHub allows up to 50 blocked-by / blocking relationships per issue.
# `first:` counts nodes toward the query's complexity, so matching the product
# limit keeps the query cheap and avoids false negatives.
_GRAPHQL_BLOCKED_BY_FIRST = 50

# Fractional jitter applied to each retry backoff (e.g. 0.25 => +/- 25%).
_JITTER_FRACTION = 0.25

# Parse "owner/repo" out of common git remote URL shapes. Intentionally loose:
# it matches the tail `.../owner/repo(.git)?` of https/ssh/git URLs, including
# `https://token@host/owner/repo.git` and `git@github.com:owner/repo.git`.
_GIT_REMOTE_URL_RE = re.compile(
    r"[:/](?P<owner>[^/\s]+)/(?P<name>[^/\s]+?)(?:\.git)?$",
    re.IGNORECASE,
)


def _parse_git_remote_url(url: str) -> tuple[str, str] | None:
    """Return (owner, repo) parsed from a git remote URL, or None if unparseable."""
    url = url.strip()
    match = _GIT_REMOTE_URL_RE.search(url)
    if not match:
        return None
    owner = match.group("owner").strip()
    name = match.group("name").strip()
    if not owner or not name:
        return None
    return owner, name


# Module-level constants for gh --json field lists.
# These are the single source of truth for all JSON field queries to GitHub.
# All call sites must use these constants — no inline field-list literals.
ISSUE_LIST_FIELDS = "number,title,url,body,labels,author,createdAt,updatedAt,state"
ISSUE_VIEW_FIELDS = (
    "number,title,url,body,labels,assignees,author,comments,createdAt,updatedAt,state"
)
PR_LIST_FIELDS = "number,title,url,headRefName,baseRefName,body,isDraft,labels,author,updatedAt,reviewDecision,statusCheckRollup,headRefOid,isCrossRepository,mergeStateStatus,mergeable,state"
PR_VIEW_FIELDS = "number,title,url,headRefName,baseRefName,body,isDraft,labels,author,updatedAt,reviewDecision,statusCheckRollup,state,mergeable,additions,deletions,headRefOid,isCrossRepository,mergeStateStatus"
PR_VIEW_MERGED_FIELDS = "state,mergedAt,headRefOid"
# The field contract for every merged-PR listing. Two producers must satisfy
# it identically: merged_prs_for_issue() queries these fields directly, and
# merged_pr_list() goes through the REST endpoint and must reproduce this exact
# key set via _normalize_rest_pr() (enforced by
# test_normalize_rest_pr_satisfies_merged_pr_list_field_contract).
#
# Consumers: workflow._merged_pr_referenced_issue_numbers() (via
# linked_issue_number()/issue_numbers_mentioned_by_pr()) reads the identity and
# branch fields; post-merge audit paths additionally need `headRefOid` to tell
# *which commit* was merged, not merely that a merge happened.
#
# Deliberately narrower than PR_LIST_FIELDS: merged PRs don't need current
# CI/review/label state, and `statusCheckRollup` in particular forces gh's
# GraphQL query to walk each PR's check-run connection — expensive across up to
# 500 merged PRs and the cause of intermittent gateway 502s on this query
# (issue #361). `headRefOid` carries no such cost: it is a scalar on the PR
# object, and on the REST path it is already present in the payload as
# head.sha, so adding it costs neither an extra request nor a graph walk.
MERGED_PR_LIST_FIELDS = "number,title,body,headRefName,isCrossRepository,state,headRefOid"
# Fields the REST normalizer emits BEYOND MERGED_PR_LIST_FIELDS. These cannot
# join the constant: it doubles as the literal `gh pr list --json` field list,
# and gh has no `mergeCommitOid` spelling (only the `mergeCommit` object), so
# adding it there would break the gh query in merged_prs_for_issue(). That is
# safe only because every consumer of these extras reads merged_pr_list(),
# which is REST-only by construction, and treats an absent key as "cannot
# verify" (the #1194 queue-sync predicate fails closed without it). Adding an
# entry here means accepting that the gh-backed path will never carry it.
MERGED_PR_REST_ONLY_FIELDS = ("mergeCommitOid",)
# Fields needed by `charlie closing-keyword-check` (issue #790): the gate only
# scans PR body/title text for closing keywords and resolves the PR's own
# declared-target binding via linked_issue_number(), which reads headRefName
# and is_cross_repository. It touches no CI/review/label state at all, so it
# must not go through the general-purpose PR_VIEW_FIELDS -- that list's
# `statusCheckRollup` forces gh's GraphQL query to walk the PR's check-run
# connection, which the default Actions GITHUB_TOKEN cannot read by default.
# This surfaced twice on the same branch, each a step deeper into the same
# query: run 30607061237 ("repository.pullRequest" itself inaccessible,
# fixed by granting `pull-requests: read`), then run 30609781476
# ("...statusCheckRollup.nodes.0.commit.statusCheckRollup" inaccessible one
# level further in, before `checks: read` had been granted at all). Rather
# than keep granting one nested-connection scope at a time and re-running to
# find the next one, the fix is at the query layer: the gate never needed
# statusCheckRollup in the first place, so a narrow field list sidesteps the
# whole class of integration-context permission gaps instead of chasing them
# field by field.
CLOSING_KEYWORD_PR_FIELDS = "title,body,headRefName,isCrossRepository"
# Fields for the post-create closing-reference verification (cw#1263): the
# only field needed is GitHub's own GraphQL resolution of which issues this
# PR will close on merge -- as opposed to `linked_issue_number`'s regex-based
# guess, `closingIssuesReferences` is GitHub's authoritative answer. Kept as
# narrow as `CLOSING_KEYWORD_PR_FIELDS` for the same reason: no CI/review
# state is needed, so no `statusCheckRollup` token-scope risk.
PR_CLOSING_ISSUES_FIELDS = "closingIssuesReferences"
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
# Minimal field lists for drift detection (reconcile.py)
# headRefOid is a plain scalar (like state/title) -- NOT a per-item graph walk
# like statusCheckRollup (see the PR_CHECKS_FIELDS note above and issue #361);
# safe to include unconditionally. Needed by detect_aviator_stale_blocked's
# commit_check_runs(sha) lookup.
RECONCILE_PR_FIELDS = "number,title,url,headRefName,baseRefName,body,state,labels,isCrossRepository,headRefOid,closedAt"
RECONCILE_ISSUE_FIELDS = "number,title,url,body,labels,state"
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


class GitHubNotFoundError(GitHubError):
    """The referenced GitHub object does not exist in this repository.

    Permanent (not retryable): raised when gh reports a GraphQL
    could-not-resolve or REST 404 for the requested object. Callers that
    derive object numbers from untrusted inputs (e.g. PR branch names) use
    this to distinguish "will never succeed" from transient gh failures.
    """


class GraphQLBudgetError(GitHubError):
    """Raised when the GitHub GraphQL rate-limit budget is too low to start a
    quota-heavy phase safely.

    Carries the remaining quota, the unix timestamp when the quota resets, and
    the configured threshold that was not met so callers can surface them in
    skip events and digests.
    """

    def __init__(self, remaining: int, reset_at: int | None, threshold: int) -> None:
        self.remaining = remaining
        self.reset_at = reset_at
        self.threshold = threshold
        super().__init__(
            f"GraphQL rate limit remaining ({remaining}) is below configured "
            f"threshold ({threshold}); reset at {reset_at}"
        )


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


class MergedPRSearchResult(list):
    """List-like result from ``merged_prs_for_issue`` with an ``ok`` flag.

    Behaves like a normal list so existing list-consuming callers keep working,
    but exposes ``ok`` so callers can distinguish a successful empty search from
    a failed ``gh pr list --search`` call (rate limit, search error, etc.).
    """

    def __init__(self, items: list[Any], ok: bool = True) -> None:
        super().__init__(items)
        self.ok = ok


_MergedPRSearchResult = MergedPRSearchResult


# Matches the job-id segment of a GitHub Actions check link, e.g.
# https://github.com/OWNER/REPO/actions/runs/RUN_ID/job/JOB_ID (optionally
# followed by a query string or #fragment, e.g. "?check_suite_focus=true").
_ACTIONS_JOB_LINK_RE = re.compile(r"/actions/runs/\d+/job/(\d+)")

# Matches the PR-number segment of a pull-request URL, e.g.
# https://github.com/OWNER/REPO/pull/123
_PR_URL_RE = re.compile(r"/pull/(\d+)")


def _pr_number_from_url(output: str) -> int | None:
    """Extract a PR number from ``gh pr create`` output.

    ``gh pr create`` prints the created PR's URL on stdout; it has no ``--json``
    flag, so this is the only machine-readable channel it offers.

    The *last* match wins, not the first. ``gh`` may precede the URL with
    progress chatter ("Creating pull request for X into main in OWNER/REPO"),
    and a caller-supplied title or body echoed into that preamble could contain
    a ``/pull/N`` link of its own -- a PR body that says "supersedes
    .../pull/900" is ordinary. The URL gh appends last is the one it created.

    Never raises; returns None when no PR URL is present.
    """
    match = None
    for match in _PR_URL_RE.finditer(output):  # noqa: B007 - last match wins
        pass
    return int(match.group(1)) if match is not None else None


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
    runtime: RuntimeConfig | None = None

    def _max_retries(self) -> int:
        if self.runtime is not None:
            return self.runtime.gh_max_retries
        return _DEFAULT_GH_MAX_RETRIES

    def _retry_base_seconds(self) -> float:
        if self.runtime is not None:
            return self.runtime.gh_retry_base_seconds
        return _DEFAULT_GH_RETRY_BASE_SECONDS

    def _timeout_seconds(self) -> float:
        if self.runtime is not None:
            return self.runtime.gh_timeout_seconds
        return _DEFAULT_GH_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        # Cache expensive list results within a single orchestrator pass to
        # avoid repeated GraphQL calls. NOT valid across passes: long-running
        # processes (charlie fleet supervise) reuse one GitHub instance for
        # many passes, so each pass must call invalidate_list_cache() or newly
        # filed issues and freshly opened/merged PRs stay invisible until the
        # process restarts.
        object.__setattr__(self, "_list_cache", {})
        # Capability collaborators (Track 2, issue #1585, design doc
        # Section 3.3): each is constructed with a back-reference to this
        # instance and reached through the delegates _install_delegates()
        # installs on the class below. Built from the same _COLLABORATORS
        # registry _ROUTES is derived from, so adding a cluster only touches
        # that one registry, not this loop.
        for collab_attr, collab_cls in _COLLABORATORS:
            object.__setattr__(self, collab_attr, collab_cls(self))

    def invalidate_list_cache(self) -> None:
        """Drop cached list results so the next call refetches from GitHub.

        Called at the start of every orchestrator pass (``loop()``); the
        cache dedupes list calls within one pass, never across passes.
        """
        self._list_cache.clear()

    def _normalize_rest_pr(self, pr: dict[str, Any]) -> dict[str, Any]:
        """Map a PR object from the REST pulls endpoint to the shape expected
        by consumers of merged_pr_list().
        """
        head = pr.get("head") or {}
        base = pr.get("base") or {}
        head_repo = (head.get("repo") or {}).get("full_name")
        base_repo = (base.get("repo") or {}).get("full_name")
        if head_repo is None or base_repo is None:
            is_cross_repository: bool | None = None
        else:
            is_cross_repository = head_repo != base_repo
        return {
            "number": pr.get("number"),
            "title": pr.get("title"),
            "body": pr.get("body"),
            "headRefName": head.get("ref"),
            "isCrossRepository": is_cross_repository,
            "state": "MERGED",
            # REST spells the head OID `head.sha`; consumers expect gh's
            # GraphQL name. Without this mapping every consumer reading
            # headRefOid off a merged PR silently sees None.
            "headRefOid": head.get("sha"),
            # Issue #1194: the merge commit that landed this PR on the base
            # branch. Its FIRST parent is the base tip immediately before
            # this merge — the only post-merge anchor from which "was this
            # content already on main?" can still be answered, since after
            # the merge everything the PR carried is main-reachable through
            # the merge commit itself.
            "mergeCommitOid": pr.get("merge_commit_sha"),
        }

    def run(
        self, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> Any:
        command = ["gh", *args]
        if self.dry_run and _is_mutating(args):
            return [] if json_output else "DRY-RUN: " + " ".join(command)

        is_mutating = _is_mutating(args)
        max_retries = self._max_retries()
        base_delay = self._retry_base_seconds()
        timeout_seconds = self._timeout_seconds()
        last_result: subprocess.CompletedProcess[str] | None = None

        for attempt in range(max_retries + 1):
            try:
                result = subprocess.run(
                    command,
                    cwd=self.repo_root,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                    check=False,
                    timeout=timeout_seconds,
                    **no_console_window_kwargs(),
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
            except subprocess.TimeoutExpired as exc:
                timeout_error = (
                    f"gh command timed out after {timeout_seconds:g}s: {' '.join(command)}"
                )
                # A timeout is not evidence about whether GitHub received the
                # request. Reads are idempotent, so they may retry. A mutation
                # that timed out may already have been applied server-side, so
                # retrying it risks double-merging, double-labelling, or a
                # duplicate comment. That is the same rule _should_retry()
                # applies to mutations, reached through a different signal —
                # checked explicitly here rather than by calling _should_retry(),
                # which classifies stderr from a process that actually returned
                # and has no string to classify for a call that never did.
                if is_mutating or attempt >= max_retries:
                    if not allow_failure:
                        raise GitHubError(timeout_error) from exc
                    return GitHubRunResult(
                        ok=False,
                        returncode=_TIMEOUT_RETURNCODE,
                        # Partial output captured before the kill. Coerced
                        # rather than trusted: TimeoutExpired.stdout is bytes
                        # when the child was not opened in text mode, and this
                        # error path must not raise on the way out.
                        stdout=exc.stdout if isinstance(exc.stdout, str) else "",
                        stderr=timeout_error,
                        value=None,
                        error=timeout_error,
                    )
                delay = base_delay * (2**attempt)
                jitter = random.uniform(-_JITTER_FRACTION * delay, _JITTER_FRACTION * delay)
                sleep_seconds = max(0.0, delay + jitter)
                logger.warning(
                    "gh command timed out after %gs (attempt %d/%d): %s; retrying in %.2fs",
                    timeout_seconds,
                    attempt + 1,
                    max_retries + 1,
                    " ".join(command),
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
                continue

            last_result = result
            output = result.stdout.strip()

            if result.returncode == 0:
                # Success path: parse and return exactly as before.
                if not allow_failure:
                    if not json_output:
                        return output
                    if not output:
                        # gh exited 0 with empty stdout: cannot distinguish an
                        # empty-but-legitimate result from an unreadable one.
                        # Callers of this path (allow_failure=False) already
                        # handle GitHubError from the retry-exhausted branch
                        # below on every call site; treat this the same way
                        # rather than silently coercing to None, which callers
                        # doing `result if isinstance(result, list) else []`
                        # (or dict equivalents) would read as "genuinely
                        # empty" (issue #756). Not retried here — the next
                        # orchestrator loop pass is the retry.
                        raise GitHubError(
                            f"gh exited 0 with empty stdout for command: {' '.join(command)}; "
                            "cannot distinguish an empty result from an unreadable one"
                        )
                    try:
                        return json.loads(output)
                    except json.JSONDecodeError as exc:
                        raise GitHubError(
                            f"Expected JSON from gh command: {' '.join(command)}"
                        ) from exc

                # allow_failure=True: always return a structured result so callers can
                # distinguish command failure from empty-but-legitimate output.
                value: Any | None = None
                if not json_output:
                    value = output
                elif output:
                    try:
                        value = json.loads(output)
                    except json.JSONDecodeError:
                        return GitHubRunResult(
                            ok=False,
                            returncode=result.returncode,
                            stdout=result.stdout,
                            stderr=result.stderr,
                            value=None,
                            error=f"Expected JSON from gh command: {' '.join(command)}",
                        )
                return GitHubRunResult(
                    ok=True,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    value=value,
                    error=None,
                )

            # Failure path: classify and either retry or surface the error.
            error = (
                result.stderr.strip() or result.stdout.strip() or f"gh exited {result.returncode}"
            )
            if attempt >= max_retries or not _should_retry(args, error, is_mutating):
                break

            delay = base_delay * (2**attempt)
            jitter = random.uniform(-_JITTER_FRACTION * delay, _JITTER_FRACTION * delay)
            sleep_seconds = max(0.0, delay + jitter)
            logger.warning(
                "Transient GitHub error (attempt %d/%d, %s): %s; retrying in %.2fs",
                attempt + 1,
                max_retries + 1,
                "read/idempotent" if not is_mutating else "mutation pre-connection",
                error,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

        # Exhausted retries or terminal failure. Reconstruct the original
        # failure contract so callers see identical behaviour for terminal errors.
        assert last_result is not None
        final_error = (
            last_result.stderr.strip() or last_result.stdout.strip() or str(last_result.returncode)
        )
        if not allow_failure:
            if _is_not_found_gh_error(final_error):
                raise GitHubNotFoundError(final_error)
            raise GitHubError(final_error)

        value = None
        error = final_error
        output = last_result.stdout.strip()
        if not json_output:
            value = output if last_result.returncode == 0 else None
        elif output:
            try:
                value = json.loads(output)
            except json.JSONDecodeError:
                error = f"Expected JSON from gh command: {' '.join(command)}"
                value = None
        return GitHubRunResult(
            ok=False,
            returncode=last_result.returncode,
            stdout=last_result.stdout,
            stderr=last_result.stderr,
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

        Returns the new PR number, or ``None`` if creation failed. Errors are
        returned as values, never raised.

        ``gh pr create`` has no ``--json`` flag -- unlike ``gh pr view``/``list``,
        it is a mutation and reports the created PR by printing its URL. Passing
        ``--json number`` made ``gh`` exit non-zero at argument parsing
        ("unknown flag: --json") *before* contacting the API, so this method
        could never succeed and no PR was ever created. It failed in the most
        expensive possible way: the caller's error string said "gh pr create
        failed", which reads as a rejection by GitHub, so the natural next step
        was to investigate permissions and branch state rather than the command
        we sent. The number is therefore parsed out of the URL, which is the
        only channel this subcommand offers.
        """
        if self.dry_run:
            return 0
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
            ],
            allow_failure=True,
        )
        if not result.ok:
            # Logged here rather than left to the caller: the caller sees only
            # ``None`` and cannot say whether gh was missing, unauthenticated,
            # rejected by the API, or handed a bad flag -- the ambiguity that
            # hid this bug.
            logger.warning(
                "gh pr create failed (head=%s base=%s rc=%s): %s",
                head,
                base,
                result.returncode,
                (result.stderr or "").strip()[:500] or "(no stderr)",
            )
            return None
        number = _pr_number_from_url(str(result.value or ""))
        if number is None:
            logger.warning(
                "gh pr create reported success for head=%s but no PR URL was found "
                "in its output: %r",
                head,
                str(result.value or "")[:500],
            )
        return number

    def _run_bool(self, args: list[str]) -> bool:
        """Run a gh command and return True iff returncode == 0.

        This is a private helper for label operations that need boolean success
        semantics without inferring from stdout/stderr string shape. Never raises
        — failures are returned as False (allow_failure semantics). Dry-run mode
        returns True (the operation would succeed if not for dry-run).
        """
        if self.dry_run and _is_mutating(args):
            return True
        result = self.run(args, allow_failure=True)
        return result.ok

    def check_graphql_rate_limit(
        self, threshold: int = _DEFAULT_GRAPHQL_RATE_LIMIT_THRESHOLD
    ) -> tuple[bool, int, int | None]:
        """Return (sufficient, remaining, reset_at) from ``gh api rate_limit``.

        Uses the REST ``rate_limit`` endpoint to inspect
        ``resources.graphql.remaining`` before starting a quota-heavy phase.
        If the endpoint cannot be reached or the response is malformed, the
        guard defaults to ``sufficient=True`` so a transient check failure does
        not wedge the fleet; callers that need strict enforcement raise
        ``GraphQLBudgetError`` when this returns ``sufficient=False``.
        """
        result = self.run(["api", "rate_limit"], json_output=True, allow_failure=True)
        data: dict[str, Any] | None = None
        if isinstance(result, GitHubRunResult):
            if not result.ok or not isinstance(result.value, dict):
                return (True, 0, None)
            data = result.value
        elif isinstance(result, dict):
            data = result
        else:
            return (True, 0, None)

        resources = data.get("resources")
        if not isinstance(resources, dict):
            return (True, 0, None)
        graphql = resources.get("graphql")
        if not isinstance(graphql, dict):
            return (True, 0, None)

        try:
            remaining = int(graphql.get("remaining", 0))
            reset_at = graphql.get("reset")
            reset_at = int(reset_at) if reset_at is not None else None
        except (TypeError, ValueError):
            return (True, 0, None)

        return (remaining >= threshold, remaining, reset_at)

    def _list_json(self, args: list[str], *, limit: int, kind: str) -> list[dict[str, Any]]:
        # run() now applies the fleet-wide bounded retry policy for transient
        # failures, so _list_json no longer needs its own ad-hoc retry loop.
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
        # Normalize labels for caching and arg building; support legacy str signature.
        if isinstance(labels, str):
            label_tuple = (labels,)
        elif labels is None:
            label_tuple = ()
        else:
            label_tuple = tuple(labels)
        effective_state = state or "open"
        cache_key = ("issue_list", effective_state, label_tuple)
        cached = self._list_cache.get(cache_key)
        if cached is not None:
            return cached

        args = [
            "issue",
            "list",
            "--limit",
            str(_LIST_LIMIT),
            "--state",
            effective_state,
            "--json",
            ISSUE_LIST_FIELDS,
        ]
        for label in label_tuple:
            args.extend(["--label", label])

        label_str = ", ".join(label_tuple) if label_tuple else "all"
        result = self._list_json(
            args,
            limit=_LIST_LIMIT,
            kind=f"issues (labels={label_str}, state={effective_state})",
        )
        self._list_cache[cache_key] = result
        return result

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
        cache_key = ("pr_list",)
        cached = self._list_cache.get(cache_key)
        if cached is not None:
            return cached
        result = self._list_json(
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
        self._list_cache[cache_key] = result
        return result

    def merged_pr_list(self) -> list[dict[str, Any]]:
        """List recently merged PRs using the REST API to avoid the expensive
        GraphQL query that gh pr list --state merged issues.

        Paginates through closed PRs (most recently updated first) and filters
        to merged PRs, returning up to _LIST_LIMIT (500) items.
        """
        cache_key = ("merged_pr_list",)
        cached = self._list_cache.get(cache_key)
        if cached is not None:
            return cached

        merged: list[dict[str, Any]] = []
        max_pages = (_LIST_LIMIT // 100) + 1
        for page in range(1, max_pages + 1):
            result = self.run(
                [
                    "api",
                    f"repos/{{owner}}/{{repo}}/pulls?state=closed&sort=updated&direction=desc&per_page=100&page={page}",
                ],
                json_output=True,
            )
            # run() returns None when gh exits 0 with empty stdout. A genuine
            # empty page comes back as the JSON array ``[]`` (a list), so a
            # non-list result means the call produced nothing parseable — an
            # unusable response, not "no more results". Silently coercing that
            # to [] makes an empty fetch indistinguishable from a failed one,
            # which would arm consumers like the #502 post-merge tripwire with
            # an empty baseline and leave them permanently blind to the merges
            # they never saw. Surface it as an error so callers can tell
            # "fetch failed" from "fetch succeeded and found nothing" (#633).
            if not isinstance(result, list):
                raise GitHubError(
                    "merged_pr_list: gh api returned no parseable list for "
                    f"page {page}; cannot distinguish empty result from unusable response"
                )
            page_prs = result
            if not page_prs:
                break
            for pr in page_prs:
                if pr.get("merged_at"):
                    merged.append(self._normalize_rest_pr(pr))
                if len(merged) >= _LIST_LIMIT:
                    break
            if len(merged) >= _LIST_LIMIT:
                break

        self._list_cache[cache_key] = merged
        return merged

    def merged_prs_for_issue(
        self,
        issue_number: int,
        branch_prefix: str,
    ) -> MergedPRSearchResult:
        """Return merged PRs that hijack-safely bind to ``issue_number``.

        Uses ``gh pr list --state merged --search`` so PRs merged long ago
        (outside the most-recent 500 window used by ``merged_pr_list``) are
        still discoverable.  The search is scoped to the issue number, so a
        single merged PR outside the global window can be finalized without
        fetching every merged PR.

        Returns a list-like object because multiple merged PRs can reference the
        same issue; callers treat any returned PR as evidence the issue is done.
        The returned object's ``ok`` flag is False when the search call itself
        failed (e.g. rate limit), allowing callers to implement circuit breakers.
        """
        query = f'"#{issue_number}"'
        result = self.run(
            [
                "pr",
                "list",
                "--state",
                "merged",
                "--search",
                query,
                "--limit",
                "20",
                "--json",
                MERGED_PR_LIST_FIELDS,
            ],
            json_output=True,
            allow_failure=True,
        )
        if isinstance(result, GitHubRunResult):
            if not result.ok:
                logger.warning(
                    "Failed to search merged PRs for issue #%d: %s",
                    issue_number,
                    result.error,
                )
                return MergedPRSearchResult([], ok=False)
            items = result.value if isinstance(result.value, list) else []
        else:
            items = result if isinstance(result, list) else []

        matched: list[dict[str, Any]] = []
        for pr in items:
            if str(pr.get("state") or "").upper() != "MERGED":
                continue
            bound = linked_issue_number(
                pr,
                is_cross_repository=pr.get("isCrossRepository"),
                branch_prefix=branch_prefix,
            )
            if bound == issue_number:
                matched.append(pr)
        return MergedPRSearchResult(matched, ok=True)

    def pr_view(self, number: int, *, fields: str = PR_VIEW_FIELDS) -> dict[str, Any]:
        """Fetch a PR via ``gh pr view --json <fields>``.

        ``fields`` defaults to the general-purpose ``PR_VIEW_FIELDS`` (CI/review/
        label state included). Callers that only need a narrow slice -- e.g.
        the closing-keyword-check gate, which never touches CI status and
        must not risk `statusCheckRollup`'s token-scope failure (see
        `CLOSING_KEYWORD_PR_FIELDS`) -- should pass their own narrower field
        list rather than filtering the wide result after the fact, so the gh
        invocation itself never requests a field it doesn't need.
        """
        result = self.run(
            [
                "pr",
                "view",
                str(number),
                "--json",
                fields,
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
                # gh pr checks ALSO exits non-zero -- with empty stdout, so
                # result.value is None here too -- when the PR simply has no
                # checks reported yet (issue #846, measured against this repo:
                # `gh pr checks 700 ...` -> exit 1, stderr "no checks reported
                # on the '...' branch", no JSON). That is indistinguishable
                # from a genuine command failure (unsupported JSON field,
                # GraphQL error, transient outage) using result.ok/result.value
                # alone, so disambiguate with a second, different endpoint
                # rather than guessing from the exit code or stderr text.
                fallback = self._pr_checks_fallback(number)
                if fallback is None:
                    # Fallback also failed (or returned an unmappable shape):
                    # genuine unavailability. Preserves pr_checks' existing
                    # None contract -- callers/loop still count this as an
                    # infrastructure error.
                    return None
                if not fallback:
                    return []
                checks = fallback
        else:
            # Legacy pre-result-object fallback
            checks = result if isinstance(result, list) else []
        # gh pr checks --json has no databaseId/runId fields; derive both the
        # GitHub Actions job id and the workflow run id from "link" and inject
        # them so downstream consumers keep reading check.get("databaseId") and
        # check.get("runId") unchanged. This also normalizes the fallback path
        # above: its mapped entries carry a "link" field in the same URL shape
        # (Actions job URL), so the same regex-based derivation applies.
        return [
            {
                **check,
                "databaseId": _job_id_from_link(check.get("link")),
                "runId": _run_id_from_link(check.get("link")),
            }
            for check in checks
        ]

    def _pr_checks_fallback(self, number: int) -> list[dict[str, Any]] | None:
        """Disambiguate a ``gh pr checks`` failure via ``statusCheckRollup`` (issue #846).

        ``gh pr checks`` cannot represent "no checks reported yet" as a
        successful empty response -- it exits non-zero with empty stdout for
        that case, identically to a genuine command failure (see the caller).
        ``gh pr view --json statusCheckRollup`` CAN: it returns a clean empty
        list with exit 0 for a PR with zero checks (measured against PR #700
        in this repo: ``{"statusCheckRollup":[]}``, exit 0).

        This field carries its own risk -- it is a per-item GraphQL graph walk
        that can fail on token scope (see the PR_CHECKS_FIELDS note above and
        issue #361) -- but any failure here simply falls through to this
        function's ``None`` return, which is exactly pr_checks' pre-existing
        "unavailable" behavior. This call can never make pr_checks' result
        worse than it was before issue #846's fix, only better (turning some
        `None`s into accurate `[]`s).

        Returns:
        - ``None`` if the fallback call itself failed, or its rollup contains
          an entry this function cannot faithfully map (see below). Callers
          treat this exactly like today's "gh command failed" case.
        - ``[]`` if the rollup is empty: legitimately no checks.
        - a list of dicts shaped like ``gh pr checks --json name,state,link``
          (name/state/link only -- databaseId/runId are injected by the
          caller) if the rollup is non-empty and every entry is a GitHub
          Actions ``CheckRun``. This covers the "gh pr checks glitched
          transiently while checks do exist" case.

        Mapping notes (verified against live PRs in this repo, not assumed):
        - ``link`` <- ``detailsUrl``: both are the same Actions job URL in
          every sample (PR #679, #839).
        - ``state`` <- ``conclusion if status == "COMPLETED" else status``:
          this is the same rule gh's own `pr checks` uses internally to
          collapse CheckRun's two-field status/conclusion into one "state".
          Verified pairwise on live data: PR #679 IN_PROGRESS check has
          ``state: IN_PROGRESS`` / ``status: IN_PROGRESS, conclusion: ""``;
          its SUCCESS check has ``state: SUCCESS`` / ``status: COMPLETED,
          conclusion: SUCCESS``; PR #839's cancelled-run checks have
          ``state: CANCELLED`` / ``status: COMPLETED, conclusion: CANCELLED``.
        - ``bucket`` is intentionally NOT mapped: it is a `gh`-CLI-side
          classification (pass/fail/pending/cancel/skipping) computed from
          state, with no GraphQL equivalent to read it back from (see the
          PR_CHECKS_FIELDS note above). Every consumer that reads "bucket"
          (checks.py, workflow.py) only uses it as an `or` alternative to
          "state" (e.g. ``state == "SUCCESS" or bucket == "pass"``), never as
          an independent requirement, so an absent bucket does not change
          classification -- "state" alone still carries it correctly.
        - Any rollup entry whose ``__typename`` is not ``"CheckRun"`` (e.g. a
          ``StatusContext`` from an external, non-Actions status check) makes
          the whole call return ``None`` instead of guessing: this repo has
          no live sample of that shape's fields, and fabricating one risks
          silently inventing check state, which is exactly what issue #846
          warns against doing at this boundary.
        """
        result = self.run(
            ["pr", "view", str(number), "--json", "statusCheckRollup"],
            json_output=True,
            allow_failure=True,
        )
        if not (
            isinstance(result, GitHubRunResult) and result.ok and isinstance(result.value, dict)
        ):
            return None
        rollup = result.value.get("statusCheckRollup")
        if not isinstance(rollup, list):
            return None
        if not rollup:
            return []
        mapped: list[dict[str, Any]] = []
        for entry in rollup:
            if not isinstance(entry, dict) or entry.get("__typename") != "CheckRun":
                return None
            status = str(entry.get("status") or "")
            conclusion = str(entry.get("conclusion") or "")
            state = conclusion if status == "COMPLETED" and conclusion else status
            mapped.append(
                {
                    "name": entry.get("name"),
                    "state": state,
                    "link": entry.get("detailsUrl"),
                }
            )
        return mapped

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

    def commit(self, sha: str) -> GitHubRunResult:
        """Fetch a single commit's metadata by SHA.

        Wraps ``gh api repos/{owner}/{repo}/commits/{sha}``. Returns a
        ``GitHubRunResult`` whose ``value`` is the parsed JSON response
        (including ``parents`` and ``committer``/``commit.committer``) on
        success, or ``None`` with ``error`` set on failure. Errors are
        returned as values, never raised.

        Callers that only want the dict can use
        ``result.value if result.ok and isinstance(result.value, dict) else None``;
        callers that need the failure reason (e.g. for event payloads, issue
        #1140) read ``result.error``. Returning the full ``GitHubRunResult``
        rather than collapsing to ``None`` preserves the transport/API error
        (TLS blip vs rate limit vs auth vs 404) at the boundary that most
        needs it, consistent with this repo's errors-as-values invariant.
        """
        result = self.run(
            ["api", f"repos/{{owner}}/{{repo}}/commits/{sha}"],
            json_output=True,
            allow_failure=True,
        )
        if isinstance(result, GitHubRunResult):
            return result
        # Dry-run short-circuit or an unexpected double that returned a raw
        # value instead of a GitHubRunResult. Normalize so the contract is
        # uniform -- callers never need to branch on the return type.
        if isinstance(result, dict):
            return GitHubRunResult(ok=True, returncode=0, stdout="", stderr="", value=result)
        return GitHubRunResult(
            ok=False,
            returncode=0,
            stdout="",
            stderr="",
            value=None,
            error=f"unexpected response from gh.run: {type(result).__name__}",
        )

    def commit_check_runs(self, sha: str) -> list[dict[str, Any]] | None:
        """Fetch the GitHub Check Runs attached to a commit SHA.

        Wraps ``gh api repos/{owner}/{repo}/commits/{sha}/check-runs`` and
        returns its ``check_runs`` array, or ``None`` on failure. Distinct
        from ``pr_checks()``/``PR_CHECKS_FIELDS``: ``gh pr checks --json``
        exposes only Commit-Status-shaped fields (its ``description`` field
        is always empty for App-created Check Runs like Aviator's
        ``aviator/checks`` -- Check Runs carry their message in
        ``output.summary``/``output.title`` instead, which ``gh pr checks``
        does not surface at all). This is the only way to read that message.
        Errors are returned as values, never raised.
        """
        result = self.run(
            ["api", f"repos/{{owner}}/{{repo}}/commits/{sha}/check-runs"],
            json_output=True,
            allow_failure=True,
        )
        value = result.value if isinstance(result, GitHubRunResult) and result.ok else None
        if not isinstance(value, dict):
            return None
        check_runs = value.get("check_runs")
        return check_runs if isinstance(check_runs, list) else None

    def workflow_runs_for_head(self, head_sha: str) -> list[dict[str, Any]] | None:
        """Fetch GitHub Actions workflow runs created for a head commit SHA.

        Wraps ``gh api repos/{owner}/{repo}/actions/runs?head_sha=<sha>`` and
        returns its ``workflow_runs`` array, or ``None`` on failure. Used to
        distinguish "CI never created a run for this head" from "CI is still
        pending": ``gh pr checks``/``statusCheckRollup`` can simply omit a
        required check that never started, which is indistinguishable from
        one still queued using check data alone. An empty (non-None) list
        means the query succeeded and GitHub genuinely has zero run objects
        for this SHA. Errors are returned as values, never raised.
        """
        result = self.run(
            ["api", f"repos/{{owner}}/{{repo}}/actions/runs?head_sha={head_sha}"],
            json_output=True,
            allow_failure=True,
        )
        value = result.value if isinstance(result, GitHubRunResult) and result.ok else None
        if not isinstance(value, dict):
            return None
        runs = value.get("workflow_runs")
        return runs if isinstance(runs, list) else None

    def pr_commits(self, number: int) -> list[dict[str, Any]] | None:
        """Fetch a PR's commits via the REST ``pulls/{number}/commits`` endpoint.

        Each item's ``commit.message`` is the exact, untruncated raw commit
        message text (subject + blank line + body), matching ``git show
        --format=%B``. Deliberately NOT ``gh pr view --json commits``: that
        GraphQL field set truncates ``messageHeadline`` at a fixed length
        (~70 chars observed) and splits the remainder into ``messageBody``
        without preserving the original text — verified on PR #788's own
        commit, where the GraphQL fields split the subject line mid-word
        ("defang o" / "utbound reviewer prose...") and would silently corrupt
        the very "keyword #N" text `closing_keyword_gate` (issue #790) needs
        to scan intact. ``per_page=100`` covers every PR this codebase
        produces (worker branches are single- or few-commit); a PR with more
        commits than that is outside this project's workflow. Returns
        ``None`` on failure — errors are returned as values, never raised.
        """
        result = self.run(
            ["api", f"repos/{{owner}}/{{repo}}/pulls/{number}/commits?per_page=100"],
            json_output=True,
            allow_failure=True,
        )
        if isinstance(result, GitHubRunResult):
            return result.value if result.ok and isinstance(result.value, list) else None
        return result if isinstance(result, list) else None

    def compare(self, base: str, head: str) -> dict[str, Any] | None:
        """Compare two commits and return the comparison metadata.

        Wraps ``gh api repos/{owner}/{repo}/compare/{base}...{head}``. Returns
        the parsed JSON response, including ``base_commit`` and
        ``merge_base_commit``, or ``None`` on failure. Errors are returned as
        values, never raised.
        """
        result = self.run(
            ["api", f"repos/{{owner}}/{{repo}}/compare/{base}...{head}"],
            json_output=True,
            allow_failure=True,
        )
        if isinstance(result, GitHubRunResult):
            return result.value if result.ok and isinstance(result.value, dict) else None
        return result if isinstance(result, dict) else None

    def branch_protection(self, base: str) -> dict[str, Any] | None:
        """Return branch protection settings for ``base``, or None on failure.

        Wraps ``gh api repos/{owner}/{repo}/branches/{base}/protection``.
        Returns ``None`` on any failure -- 404 (no protection configured),
        rate limit, transient network error, gh not installed. Errors are
        returned as values, never raised.

        Cached per orchestrator pass in ``_list_cache`` (issue #812): callers
        use this to derive base-freshness policy (``required_status_checks.
        strict``) instead of a hardcoded config constant, and need exactly
        one API call per base ref per pass, not one per PR. A failed read is
        cached as ``None`` too, so a 404/rate-limit does not turn into a
        per-PR retry storm within the same pass.

        Safety note for callers: ``None`` means "could not be read", not "no
        freshness required". Any caller gating a safety check on this value
        must fail closed on ``None``.
        """
        cache_key = ("branch_protection", base)
        if cache_key in self._list_cache:
            return self._list_cache[cache_key]
        result = self.run(
            ["api", f"repos/{{owner}}/{{repo}}/branches/{base}/protection"],
            json_output=True,
            allow_failure=True,
        )
        value: dict[str, Any] | None = None
        if isinstance(result, GitHubRunResult):
            value = result.value if result.ok and isinstance(result.value, dict) else None
        elif isinstance(result, dict):
            value = result
        self._list_cache[cache_key] = value
        return value

    def compare_diff(self, base: str, head: str) -> str | None:
        """Return the plain unified-diff text between two commits (three-dot compare).

        Wraps the same ``gh api repos/{owner}/{repo}/compare/{base}...{head}``
        endpoint as :meth:`compare`, but requests the ``application/vnd.github.
        v3.diff`` media type so the response body is a ready-to-write unified
        diff (like :meth:`pr_diff`) instead of JSON compare metadata. Tolerates
        a rebased/diverged/GC'd ``base`` the same way GitHub's three-dot
        compare does. Returns ``None`` on any failure (404, API error, gh not
        installed) — errors are returned as values, never raised.
        """
        result = self.run(
            [
                "api",
                f"repos/{{owner}}/{{repo}}/compare/{base}...{head}",
                "-H",
                "Accept: application/vnd.github.v3.diff",
            ],
            allow_failure=True,
        )
        if isinstance(result, GitHubRunResult):
            return result.value if result.ok and isinstance(result.value, str) else None
        return result if isinstance(result, str) else None

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
            (
                "ISSUE_LIST_FIELDS",
                ["issue", "list", "--state", "open", "--limit", "1", "--json", probe],
                ISSUE_LIST_FIELDS,
            ),
            ("ISSUE_VIEW_FIELDS", ["issue", "view", "0", "--json", probe], ISSUE_VIEW_FIELDS),
            (
                "PR_LIST_FIELDS",
                ["pr", "list", "--state", "open", "--limit", "1", "--json", probe],
                PR_LIST_FIELDS,
            ),
            (
                "MERGED_PR_LIST_FIELDS",
                ["pr", "list", "--state", "merged", "--limit", "1", "--json", probe],
                MERGED_PR_LIST_FIELDS,
            ),
            ("PR_VIEW_FIELDS", ["pr", "view", "0", "--json", probe], PR_VIEW_FIELDS),
            ("PR_CHECKS_FIELDS", ["pr", "checks", "0", "--json", probe], PR_CHECKS_FIELDS),
            (
                "LABEL_LIST_FIELDS",
                ["label", "list", "--limit", "1", "--json", probe],
                LABEL_LIST_FIELDS,
            ),
            (
                "RECONCILE_PR_FIELDS",
                ["pr", "list", "--state", "all", "--limit", "1", "--json", probe],
                RECONCILE_PR_FIELDS,
            ),
            (
                "RECONCILE_ISSUE_FIELDS",
                ["issue", "list", "--state", "open", "--limit", "1", "--json", probe],
                RECONCILE_ISSUE_FIELDS,
            ),
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
                    timeout=self._timeout_seconds(),
                    **no_console_window_kwargs(),
                )
            except FileNotFoundError as exc:
                raise ConfigError("GitHub CLI `gh` is not installed or not on PATH") from exc
            except subprocess.TimeoutExpired as exc:
                # No retry here: this probe runs once at startup to validate
                # field lists, and a hang means the environment cannot answer
                # the question at all. Surfacing it as ConfigError matches the
                # other failure modes of this loop rather than hanging boot.
                raise ConfigError(
                    f"gh timed out validating field list {name} after {self._timeout_seconds():g}s"
                ) from exc

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

    def pr_ready(self, number: int) -> GitHubRunResult:
        """Mark a draft PR as ready for review via ``gh pr ready`` (issue #818).

        Returns a structured result so callers can distinguish success from
        failure without inferring from output shape -- errors from external
        processes come back as values here, never exceptions. Dry-run mode
        returns a synthetic ok=True result (the operation would succeed if not
        for dry-run); this mirrors ``_run_bool``'s explicit guard because
        ``.run()`` itself returns a bare string under dry-run, not a
        ``GitHubRunResult``.
        """
        args = ["pr", "ready", str(number)]
        if self.dry_run and _is_mutating(args):
            return GitHubRunResult(
                ok=True, returncode=0, stdout="", stderr="", value=None, error=None
            )
        result = self.run(args, allow_failure=True)
        assert isinstance(result, GitHubRunResult)
        return result

    def pr_close(self, number: int) -> GitHubRunResult:
        """Close a PR via ``gh pr close`` (issue #1274, W17).

        Half of the close/reopen stale-checks retrigger mechanism: closing
        and then reopening a PR is a common way to force GitHub to
        re-evaluate branch protection / re-create a check-suite run for a
        head where Actions never created one. Modeled exactly on
        ``pr_ready`` -- structured result, dry-run synthetic ok=True guard,
        never raises.
        """
        args = ["pr", "close", str(number)]
        if self.dry_run and _is_mutating(args):
            return GitHubRunResult(
                ok=True, returncode=0, stdout="", stderr="", value=None, error=None
            )
        result = self.run(args, allow_failure=True)
        assert isinstance(result, GitHubRunResult)
        return result

    def pr_reopen(self, number: int) -> GitHubRunResult:
        """Reopen a PR via ``gh pr reopen`` (issue #1274, W17).

        Paired with ``pr_close`` for the close/reopen stale-checks
        retrigger mechanism. Modeled exactly on ``pr_ready``.

        NOTE (unverified, flagged per issue #1274's binding comment item 6):
        whether reopening a PR actually causes GitHub Actions to create a
        fresh check-suite run for the PR's CURRENT head (as opposed to
        being a no-op for check-suite purposes) has not been confirmed
        against a live repository -- it cannot be verified with a real `gh`
        call inside a sandboxed/mocked test environment. The
        ``push_empty_commit`` fallback exists specifically because this is
        uncertain; an operator should confirm close/reopen's effect against
        a disposable fixture PR before relying on it as the primary
        mechanism in production.
        """
        args = ["pr", "reopen", str(number)]
        if self.dry_run and _is_mutating(args):
            return GitHubRunResult(
                ok=True, returncode=0, stdout="", stderr="", value=None, error=None
            )
        result = self.run(args, allow_failure=True)
        assert isinstance(result, GitHubRunResult)
        return result

    def push_empty_commit(self, branch: str) -> GitHubRunResult:
        """Push a content-free commit onto ``branch`` via the Git Data API
        (issue #1274, W17).

        Fallback CI-retrigger mechanism, used only when ``pr_close`` +
        ``pr_reopen`` does not mechanically succeed (either call returns
        not-ok). Moves the branch tip to a new commit that has the exact
        same tree as the current tip (i.e. no content change) so a
        push-triggered workflow re-evaluates the branch at a fresh head
        SHA, without altering any file. Four `gh api` reads/writes, in
        order:

        1. GET the branch ref to find the current tip commit SHA.
        2. GET that commit object to find its tree SHA (the new commit
           reuses it unchanged).
        3. POST a new commit object with that tree and the old tip as its
           sole parent.
        4. PATCH the branch ref to point at the new commit.

        Every step returns errors as values -- this method never raises,
        per this repo's external-process-errors-as-values convention, and
        stops at the first failing step rather than attempting later steps
        against inconsistent state. Dry-run mode returns a synthetic
        ok=True result before any `gh` call is made at all (not just before
        the final PATCH): this operation is unconditionally mutating end to
        end -- there is no read-only prefix of it worth letting a
        --dry-run caller observe, unlike e.g. ``workflow_runs_for_head``.
        """
        if self.dry_run:
            return GitHubRunResult(
                ok=True, returncode=0, stdout="", stderr="", value=None, error=None
            )

        ref_result = self.run(
            ["api", f"repos/{{owner}}/{{repo}}/git/refs/heads/{branch}"],
            json_output=True,
            allow_failure=True,
        )
        if not isinstance(ref_result, GitHubRunResult) or not ref_result.ok:
            error = (
                ref_result.error
                if isinstance(ref_result, GitHubRunResult)
                else f"unexpected response reading ref for branch {branch!r}"
            )
            return GitHubRunResult(
                ok=False,
                returncode=ref_result.returncode if isinstance(ref_result, GitHubRunResult) else 0,
                stdout="",
                stderr="",
                value=None,
                error=error or f"failed to read ref for branch {branch!r}",
            )
        ref_value = ref_result.value
        tip_sha = ref_value.get("object", {}).get("sha") if isinstance(ref_value, dict) else None
        if not isinstance(tip_sha, str) or not tip_sha:
            return GitHubRunResult(
                ok=False,
                returncode=0,
                stdout="",
                stderr="",
                value=None,
                error=f"could not determine tip SHA for branch {branch!r}",
            )

        commit_lookup = self.run(
            ["api", f"repos/{{owner}}/{{repo}}/git/commits/{tip_sha}"],
            json_output=True,
            allow_failure=True,
        )
        if not isinstance(commit_lookup, GitHubRunResult) or not commit_lookup.ok:
            error = (
                commit_lookup.error
                if isinstance(commit_lookup, GitHubRunResult)
                else f"unexpected response reading commit {tip_sha!r}"
            )
            return GitHubRunResult(
                ok=False,
                returncode=(
                    commit_lookup.returncode if isinstance(commit_lookup, GitHubRunResult) else 0
                ),
                stdout="",
                stderr="",
                value=None,
                error=error or f"failed to read commit {tip_sha!r}",
            )
        commit_value = commit_lookup.value
        tree_sha = (
            commit_value.get("tree", {}).get("sha") if isinstance(commit_value, dict) else None
        )
        if not isinstance(tree_sha, str) or not tree_sha:
            return GitHubRunResult(
                ok=False,
                returncode=0,
                stdout="",
                stderr="",
                value=None,
                error=f"could not determine tree SHA for commit {tip_sha!r}",
            )

        new_commit = self.run(
            [
                "api",
                "-X",
                "POST",
                "repos/{owner}/{repo}/git/commits",
                "-f",
                "message=chore: retrigger CI (empty commit)",
                "-f",
                f"tree={tree_sha}",
                "-f",
                f"parents[]={tip_sha}",
            ],
            json_output=True,
            allow_failure=True,
        )
        if not isinstance(new_commit, GitHubRunResult) or not new_commit.ok:
            error = (
                new_commit.error
                if isinstance(new_commit, GitHubRunResult)
                else "unexpected response creating empty commit"
            )
            return GitHubRunResult(
                ok=False,
                returncode=new_commit.returncode if isinstance(new_commit, GitHubRunResult) else 0,
                stdout="",
                stderr="",
                value=None,
                error=error or "failed to create empty commit",
            )
        new_commit_value = new_commit.value
        new_sha = new_commit_value.get("sha") if isinstance(new_commit_value, dict) else None
        if not isinstance(new_sha, str) or not new_sha:
            return GitHubRunResult(
                ok=False,
                returncode=0,
                stdout="",
                stderr="",
                value=None,
                error="empty-commit creation response had no sha",
            )

        ref_update = self.run(
            [
                "api",
                "-X",
                "PATCH",
                f"repos/{{owner}}/{{repo}}/git/refs/heads/{branch}",
                "-f",
                f"sha={new_sha}",
            ],
            json_output=True,
            allow_failure=True,
        )
        if not isinstance(ref_update, GitHubRunResult):
            return GitHubRunResult(
                ok=False,
                returncode=0,
                stdout="",
                stderr="",
                value=None,
                error=f"unexpected response updating ref for branch {branch!r}",
            )
        return ref_update

    def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
        """Check which of the given issue numbers are currently open.

        Returns a set of issue numbers that are open. Issues that don't exist
        or are closed are not included in the result. This is used for the
        dependency gate to check if blocker issues are still open.

        Per-issue-number results are cached in the pass-scoped ``_list_cache``
        (keyed ``("issue_open", number)``). Cache misses are first resolved in
        a single batched GraphQL query (one subprocess for the whole set); only
        if the batch fails do we fall back to the previous parallel
        per-``issue_view`` fetch.

        Args:
            issue_numbers: List of issue numbers to check

        Returns:
            Set of issue numbers that are currently open
        """
        if not issue_numbers:
            return set()

        open_issues: set[int] = set()
        uncached: list[int] = []
        for number in issue_numbers:
            cached = self._list_cache.get(("issue_open", number))
            if cached is None:
                uncached.append(number)
            elif cached:
                open_issues.add(number)

        if uncached:
            try:
                states = self._graphql_issue_states(uncached)
                for number, is_open in states.items():
                    self._list_cache[("issue_open", number)] = is_open
                    if is_open:
                        open_issues.add(number)
            except (GitHubError, OSError, ValueError, TypeError):
                logger.warning(
                    "Batched issue state query failed, falling back to per-issue view",
                    exc_info=True,
                )

                # Fallback to the previous parallel per-issue view fetch.
                def _fetch_state(number: int) -> tuple[int, bool]:
                    try:
                        issue = self.issue_view(number)
                        is_open = str(issue.get("state") or "").upper() == "OPEN"
                    except (GitHubError, ValueError, TypeError):
                        is_open = False
                    return number, is_open

                max_workers = min(_MAX_ISSUE_STATE_WORKERS, len(uncached))
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    for number, is_open in pool.map(_fetch_state, uncached):
                        self._list_cache[("issue_open", number)] = is_open
                        if is_open:
                            open_issues.add(number)

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

    def _repo_owner_name(self) -> tuple[str, str]:
        """Resolve the repository owner and name from the local git remote.

        Prefers `git remote get-url origin` over a network round-trip so this
        can work under `--dry-run` and so fleet status does not pay an extra
        `gh repo view` process. The result is cached in ``_list_cache`` for the
        duration of the pass.

        Raises GitHubError when the remote cannot be read or does not point at
        a parseable GitHub-style URL.
        """
        cache_key = ("_repo_owner_name",)
        cached = self._list_cache.get(cache_key)
        if isinstance(cached, tuple) and len(cached) == 2:
            return cached

        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                **no_console_window_kwargs(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise GitHubError(f"Unable to read git remote origin: {exc}") from exc

        if result.returncode != 0:
            raise GitHubError(
                f"git remote get-url origin failed: "
                f"{(result.stderr or '').strip() or result.returncode}"
            )

        parsed = _parse_git_remote_url(result.stdout.strip())
        if parsed is None:
            raise GitHubError(
                f"Unable to parse owner/name from git remote: {result.stdout.strip()[:200]}"
            )

        self._list_cache[cache_key] = parsed
        return parsed

    def _graphql_query(
        self,
        query: str,
        *,
        allow_failure: bool = False,
    ) -> dict[str, Any]:
        """Run a single read-only GraphQL query via ``gh api graphql``.

        The command is classified as read-only by ``_api_is_mutating`` because
        it starts with the GraphQL ``query`` keyword, so it is not suppressed by
        ``--dry-run``. Raises GitHubError for non-zero exit or a response that
        contains no usable ``data``.
        """
        result = self.run(
            [
                "api",
                "graphql",
                "-f",
                f"query={query}",
                "-f",
                f"owner={self._repo_owner_name()[0]}",
                "-f",
                f"name={self._repo_owner_name()[1]}",
            ],
            json_output=True,
            allow_failure=allow_failure,
        )

        if isinstance(result, GitHubRunResult):
            if not result.ok:
                raise GitHubError(f"GraphQL query failed: {result.error}")
            value = result.value
        else:
            value = result

        if not isinstance(value, dict):
            raise GitHubError("GraphQL query returned non-dict JSON")

        if value.get("data") is None and value.get("errors"):
            errors = value.get("errors")
            if isinstance(errors, list):
                messages = [str(e.get("message", e)) for e in errors]
                raise GitHubError("; ".join(messages))
            raise GitHubError(str(errors))

        return value

    def _graphql_issue_states(self, issue_numbers: list[int]) -> dict[int, bool]:
        """Fetch open/closed state for many issue numbers in one GraphQL query.

        Returns a mapping ``issue_number -> is_open`` for the requested numbers.
        Missing or errored issues are treated as not open, matching the
        contract of ``are_issues_open``.
        """
        if not issue_numbers:
            return {}

        owner, name = self._repo_owner_name()
        states: dict[int, bool] = {}

        for i in range(0, len(issue_numbers), _GRAPHQL_BATCH_SIZE):
            chunk = issue_numbers[i : i + _GRAPHQL_BATCH_SIZE]
            fields = " ".join(f"s_{n}: issue(number: {n}) {{ number state }}" for n in chunk)
            query = (
                f"query($owner: String!, $name: String!) {{ "
                f"repository(owner: $owner, name: $name) {{ {fields} }} "
                f"}}"
            )

            data = self._graphql_query(query).get("data", {})
            repo = data.get("repository", {})
            if not isinstance(repo, dict):
                raise GitHubError("GraphQL response missing repository")

            for number in chunk:
                alias = f"s_{number}"
                issue = repo.get(alias)
                if not isinstance(issue, dict):
                    states[number] = False
                    continue
                returned_number = issue.get("number")
                if returned_number is not None:
                    number = int(returned_number)
                is_open = str(issue.get("state") or "").upper() == "OPEN"
                states[number] = is_open

        return states

    def _graphql_issue_dependencies(self, issue_numbers: list[int]) -> dict[int, list[int]]:
        """Fetch GitHub-native ``blockedBy`` dependencies for many issues at once.

        Returns a mapping ``issue_number -> [blocker_number, ...]``. Also warms
        the ``("issue_open", blocker_number)`` cache for the returned blockers
        so the downstream ``are_issues_open`` call can avoid refetching them.
        """
        if not issue_numbers:
            return {}

        owner, name = self._repo_owner_name()
        deps_by_number: dict[int, list[int]] = {}

        for i in range(0, len(issue_numbers), _GRAPHQL_BATCH_SIZE):
            chunk = issue_numbers[i : i + _GRAPHQL_BATCH_SIZE]
            fields = " ".join(
                f"i_{n}: issue(number: {n}) {{ "
                f"number "
                f"blockedBy(first: {_GRAPHQL_BLOCKED_BY_FIRST}) {{ "
                f"nodes {{ number state }} "
                f"pageInfo {{ hasNextPage }} "
                f"}} "
                f"}}"
                for n in chunk
            )
            query = (
                f"query($owner: String!, $name: String!) {{ "
                f"repository(owner: $owner, name: $name) {{ {fields} }} "
                f"}}"
            )

            data = self._graphql_query(query).get("data", {})
            repo = data.get("repository", {})
            if not isinstance(repo, dict):
                raise GitHubError("GraphQL response missing repository")

            for number in chunk:
                alias = f"i_{number}"
                issue = repo.get(alias)
                if not isinstance(issue, dict):
                    deps_by_number[number] = []
                    self._list_cache[("issue_dependencies", number)] = []
                    continue

                returned_number = issue.get("number")
                if returned_number is not None:
                    number = int(returned_number)

                blocked_by: list[int] = []
                blocked_by_conn = issue.get("blockedBy")
                if isinstance(blocked_by_conn, dict):
                    if blocked_by_conn.get("pageInfo", {}).get("hasNextPage"):
                        logger.warning(
                            "Issue #%d has more than %d blockedBy entries; "
                            "only the first page was fetched",
                            number,
                            _GRAPHQL_BLOCKED_BY_FIRST,
                        )
                    for node in blocked_by_conn.get("nodes") or []:
                        if isinstance(node, dict):
                            blocker_number = node.get("number")
                            if blocker_number is not None:
                                blocked_by.append(int(blocker_number))
                                # Cache the blocker state now so
                                # are_issues_open does not need to re-derive it.
                                is_open = str(node.get("state") or "").upper() == "OPEN"
                                self._list_cache[("issue_open", int(blocker_number))] = is_open

                deps_by_number[number] = blocked_by
                self._list_cache[("issue_dependencies", number)] = blocked_by

        return deps_by_number

    def issue_dependencies(self, issue_numbers: list[int]) -> dict[int, list[int]]:
        """Fetch GitHub-native blocked-by relationships for a list of issues.

        Uses a single batched GraphQL query (or a small number of chunked
        queries for large backlogs) instead of one REST API call per issue.
        Mirrors the fail-open contract of ``get_github_issue_dependencies``:
        if the batched query cannot be built or fails, falls back to the
        original per-issue REST calls and returns whatever was successfully
        resolved.
        """
        if not issue_numbers:
            return {}

        try:
            return self._graphql_issue_dependencies(issue_numbers)
        except (GitHubError, OSError, ValueError, TypeError):
            logger.warning(
                "Batched issue dependency query failed, falling back to per-issue REST",
                exc_info=True,
            )

        # Fallback to the original per-issue REST endpoint, preserving the
        # warm-cache contract for callers that later call get_github_issue_dependencies.
        result: dict[int, list[int]] = {}
        for number in issue_numbers:
            deps = get_github_issue_dependencies(self, number)
            result[number] = deps
        return result


# Install the capability delegates now that `GitHub`'s class body is fully
# defined. `_ROUTES` is empty in L01 (every collaborator class is still
# empty), so this is a no-op: it installs nothing and `GitHub`'s lexical
# member surface is unchanged.
_install_delegates(GitHub)


@runtime_checkable
class GitHubLike(
    CommentsLike,
    LabelsLike,
    ChecksLike,
    RepoMetaLike,
    PullRequestsLike,
    IssuesLike,
    MergeBranchLike,
    Protocol,
):
    """Structural interface for the GitHub surface the orchestrator calls.

    Production functions accept ``gh: GitHubLike`` instead of the concrete
    ``GitHub`` class so test doubles can satisfy the contract structurally
    without subclassing the frozen dataclass (issue #593).

    Redeclared (Track 2, issue #1585; design doc Section 4.1) as the union of
    the seven capability sub-protocols plus the two members that stay on the
    owner (``dry_run``, ``run``). Five members below are *also* inherited
    from a sub-protocol but are redeclared directly in this body: Protocol
    inheritance puts an inherited member in the *sub-protocol's* ``__dict__``,
    not the union's, and five existing tests assert
    ``name in GitHubLike.__dict__`` by name (below). Redeclaring costs
    nothing on the member_count metric (``_is_protocol_base`` excludes
    Protocol subclasses entirely).
    """

    # Declared as a read-only property, not a settable attribute, so the
    # frozen ``GitHub`` dataclass (whose ``dry_run`` field is immutable)
    # satisfies the protocol. A plain ``dry_run: bool`` annotation would
    # require a *writable* attribute, which a frozen dataclass cannot provide
    # — that mismatch was the root cause of every ``GitHub``-vs-``GitHubLike``
    # ``reportArgumentType`` error in src/ (issue #733). Test doubles that set
    # ``self.dry_run`` in ``__init__`` still satisfy a read-only property: a
    # settable attribute is a superset of a read-only one.
    @property
    def dry_run(self) -> bool: ...

    def run(
        self, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> Any: ...

    # Redeclared directly (see class docstring): inherited from MergeBranchLike.
    def branch_protection(self, base: str) -> dict[str, Any] | None: ...

    # Redeclared directly (see class docstring): inherited from PullRequestsLike.
    def pr_ready(self, number: int) -> GitHubRunResult: ...

    # Redeclared directly (see class docstring): inherited from MergeBranchLike.
    def pr_close(self, number: int) -> GitHubRunResult: ...

    # Redeclared directly (see class docstring): inherited from MergeBranchLike.
    def pr_reopen(self, number: int) -> GitHubRunResult: ...

    # Redeclared directly (see class docstring): inherited from MergeBranchLike.
    def push_empty_commit(self, branch: str) -> GitHubRunResult: ...


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
# reference in freeform text actually links the PR to issue N. Shared between
# the matching regex below and `_CLOSING_KEYWORD_DEFANG_RE` so the two
# patterns cannot drift apart.
_CLOSING_KEYWORDS_ALT = r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)"
_CLOSING_KEYWORD_REF = re.compile(_CLOSING_KEYWORDS_ALT + r"\s+#(\d+)", flags=re.IGNORECASE)
# Rewrites `<keyword> #N` to `<keyword> issue N` — used by
# `defang_closing_keywords` to strip the auto-close/binding syntax from text
# that will be embedded in a PR body/comment charlie-work does not control
# downstream (e.g. a rework brief a worker reads and copies into its own PR).
_CLOSING_KEYWORD_DEFANG_RE = re.compile(
    r"(" + _CLOSING_KEYWORDS_ALT + r")(\s+)#(\d+)", flags=re.IGNORECASE
)
# The orchestrator's own branch convention (agent/issue-N-slug). A head ref is
# the trusted signal because the orchestrator created it at dispatch.
_BRANCH_ISSUE_REF = re.compile(r"issue[-_/](\d+)", flags=re.IGNORECASE)

# Negation words/contractions that, when found shortly before a closing
# keyword match, mean the keyword is being negated ("does not fix #649")
# rather than asserting a real closing action. Kept as a module-level
# constant — never inline literals at the match site — so the vocabulary is
# audited and extended in exactly one place.
_NEGATION_WHOLE_WORDS = ("not", "never", "without", "cannot")
# Matched as a bare substring (no leading \b): "doesn't"/"can't" have no word
# boundary immediately before the "n" in "n't" — the apostrophe is not a
# \w character, so "doesn" + "'t" is one continuous \w-run from \b's
# perspective and a \b-anchored "n't" would silently never match.
_NEGATION_CONTRACTION_SUFFIX = "n't"
_NEGATION_RE = re.compile(
    r"\b(?:" + "|".join(_NEGATION_WHOLE_WORDS) + r")\b|" + re.escape(_NEGATION_CONTRACTION_SUFFIX),
    flags=re.IGNORECASE,
)
# How many characters back to look for a negation word before a closing
# keyword. 32 comfortably covers every negation phrase in the acceptance
# criteria ("does not " = 9 chars, "without " = 8) with headroom for a short
# intervening clause. The tradeoff is deliberate and biased toward the safe
# direction: at this width, "This is not a revert. Fixes #700" is treated as
# negated even though the negation and the keyword sit in different
# sentences. A missed binding leaves the issue in its current label state
# (safe); a false binding silently marks live work done (unsafe) — so
# over-triggering the guard is acceptable, under-triggering is not.
_NEGATION_LOOKBEHIND_CHARS = 32


def _has_preceding_negation(text: str, match_start: int) -> bool:
    """True if a negation word/contraction appears shortly before match_start."""
    window_start = max(0, match_start - _NEGATION_LOOKBEHIND_CHARS)
    # Pass pos/endpos (not a string slice) so \b at the window edge is still
    # resolved against the real surrounding text, not an artificial cut.
    return bool(_NEGATION_RE.search(text, window_start, match_start))


def iter_unnegated_closing_keyword_matches(text: str) -> Iterator[re.Match[str]]:
    """Yield every `_CLOSING_KEYWORD_REF` match in ``text`` not preceded by negation.

    This is the shared core scanning primitive (finditer over every
    keyword+``#N`` occurrence, filtered by the negation lookback) behind both
    consumers that need it:

    - `_first_unnegated_closing_keyword_match` (below) takes only the first —
      `linked_issue_number`'s label-transition binding only ever needs one
      match to bind an issue.
    - `closing_keyword_gate.find_unexpected_closing_references` (issue #790)
      needs *every* match across a whole PR body plus every commit message,
      because GitHub's native auto-close-on-merge scans all of those
      surfaces for every closing-keyword reference, not just the first.

    Both consume this one generator rather than each hand-rolling their own
    `finditer` + negation-lookback scan, so the two callers cannot drift
    apart on what counts as a live closing reference.
    """
    for match in _CLOSING_KEYWORD_REF.finditer(text):
        if not _has_preceding_negation(text, match.start()):
            yield match


def _first_unnegated_closing_keyword_match(text: str) -> re.Match[str] | None:
    """Return the first `_CLOSING_KEYWORD_REF` match not preceded by negation.

    A negated match (e.g. "does not fix #649") must not shadow a later,
    genuine match in the same field (e.g. "...but this PR also fixes #700") —
    scanning continues past it instead of giving up on the whole field.
    """
    return next(iter_unnegated_closing_keyword_matches(text), None)


def defang_closing_keywords(text: str) -> str:
    """Rewrite `<keyword> #N` to `<keyword> issue N` in freeform text.

    Used to sanitize text that charlie-work writes into a PR body, comment,
    or rework brief that a downstream reader (GitHub's auto-close, or a
    worker agent copying reviewer prose into its own PR) does not go through
    `linked_issue_number`'s hijack-safety checks. The issue number stays
    legible to a human; only the syntax that triggers a live closing
    reference or label-transition binding is removed. Unconditional — unlike
    the negation guard above, this rewrites every keyword match regardless
    of surrounding negation, since the goal here is to remove the trigger
    syntax entirely, not to judge intent.
    """
    return _CLOSING_KEYWORD_DEFANG_RE.sub(r"\g<1>\g<2>issue \g<3>", text)


def build_branch_issue_validator(
    gh: GitHubLike,
) -> Callable[[int], bool] | None:
    """Build a validator for branch-name-derived issue numbers (issue #1229).

    This is the single-point-of-enforcement constructor for the
    ``branch_issue_validator`` callable consumed by ``linked_issue_number``.
    Every call site that resolves a branch-name issue number against the real
    open-issue set -- the module-level sweeps
    (``_detect_and_handle_orphaned_workers``,
    ``_classify_dead_sessions_and_update_throttle_state``), the rework-routing
    ``OrchestratorApp`` methods, and the dispatch-claim ``pr_by_issue``
    construction -- routes through here so the open-issue fetch, failure
    handling, and ``_LIST_LIMIT`` tradeoff cannot diverge between call
    surfaces.

    Returns a callable that returns True iff the given number corresponds to
    a real *open* issue in this repo, or None when the open-issue list cannot
    be fetched (API outage). Callers that receive None should pass None to
    ``linked_issue_number``'s ``branch_issue_validator`` -- the function then
    trusts the branch-name binding unconditionally, preserving the pre-#1229
    behavior rather than blocking the sweep during a transient GitHub
    failure.

    ``issue_list(state="open")`` is cached within a pass on the real
    ``GitHub`` client, so repeated calls to this helper in the same pass
    share a single GitHub API call. The list is capped at ``_LIST_LIMIT``
    (500); a repo with more open issues than the cap could see a false
    negative (a genuinely open issue treated as absent), which is the safe
    direction -- refusing a branch-name binding never corrupts state, it
    only defers an issue-adjacent operation until the issue is confirmed by
    a closing keyword.
    """
    try:
        open_issues = gh.issue_list(state="open")
    except Exception:
        # GitHubError (API outage), AttributeError (test fakes without
        # issue_list), or any other transient failure -- the safe direction
        # is to skip validation (return None) so callers preserve the
        # pre-#1229 branch-name trust behavior rather than crashing or
        # blocking the sweep.
        return None
    return build_branch_issue_validator_from_issues(open_issues)


def build_branch_issue_validator_from_issues(
    open_issues: Iterable[dict[str, Any]],
) -> Callable[[int], bool]:
    """Build a branch-issue validator from a pre-fetched OPEN issue snapshot.

    This is the single construction path for the open-number set that
    ``build_branch_issue_validator`` (which fetches via
    ``issue_list(state="open")``) and any caller that already holds an
    open-issue snapshot share, so the ``int(number)`` extraction and the
    ``frozenset`` shape cannot diverge between call surfaces.

    Use this instead of ``build_branch_issue_validator`` when the caller has
    already fetched the issue list in the same pass (e.g. ``reconcile.detect_drift``
    fetches ``issues?state=all`` as one of its two ``gh.run`` list queries and
    must not issue a third). Unlike ``build_branch_issue_validator``, this
    never returns None: the snapshot is already in hand, so there is no
    fetch-outage fail-open path -- validation always runs. ``open_issues``
    must already be filtered to OPEN state by the caller (this helper only
    extracts numbers, it does not re-filter by state).
    """
    open_numbers = frozenset(int(i["number"]) for i in open_issues if i.get("number") is not None)
    return lambda n: n in open_numbers


def linked_issue_number(
    pr: dict[str, Any],
    *,
    is_cross_repository: bool | None,
    branch_prefix: str,
    branch_issue_validator: Callable[[int], bool] | None = None,
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

    A closing keyword preceded by a negation ("does not fix #649") also does
    not bind — see `_first_unnegated_closing_keyword_match`. This prevents a
    false LABEL TRANSITION in charlie-work's own state machine; it has no
    effect on GitHub's own issue auto-close, which is a separate mechanism
    charlie-work does not control.

    Issue #1229: ``branch_issue_validator``, when provided, is called with
    the candidate issue number parsed from the branch name. If it returns
    False (the number does not correspond to a real open issue), the
    branch-name binding is rejected and the function falls through to the
    closing-keyword path instead. This prevents a stale branch-name number
    (e.g. a branch ``agent/issue-709-…`` left over from a merged issue/PR
    #709, reused by an unrelated issue-less PR) from silently keying a
    rework episode under ``state["issues"]["709"]`` and colliding with the
    unrelated issue's lifecycle. When the validator is None (the default),
    the branch-name binding is trusted unconditionally — preserving the
    behavior of callers that do not need the validation.
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
        # 3. (Issue #1229) The parsed number is a real open issue, when a
        #    validator is supplied. Without a validator, trust unconditionally
        #    to preserve existing caller behavior.
        has_correct_prefix = head.startswith(branch_prefix)
        if has_correct_prefix:
            candidate = int(match.group(1))
            if branch_issue_validator is None or branch_issue_validator(candidate):
                return candidate
            # Branch-name number is stale/unmatched — fall through to the
            # closing-keyword path rather than binding to a non-existent or
            # closed issue.
    # For same-repo PRs, trust closing keywords in title/body — but a
    # negated keyword ("does not fix #649") must not bind; see
    # `_first_unnegated_closing_keyword_match`.
    for text in (str(pr.get("title") or ""), str(pr.get("body") or "")):
        match = _first_unnegated_closing_keyword_match(text)
        if match:
            return int(match.group(1))
    return None


# Explicit issue-reference pattern: the word "issue" or "issues" followed by
# an optional space and then "#N".  This is deliberately narrower than a bare
# ``#N`` so a merged PR that mentions a different PR (e.g. "PR #181") is not
# mistaken for an issue reference.  Closing-keyword binding is handled by
# ``linked_issue_number``.
_ISSUE_MENTION_RE = re.compile(r"\b(?:issue|issues)\s*#(\d+)\b", flags=re.IGNORECASE)
# Stripped before matching to cut two concrete false-positive classes: a
# fenced code sample that happens to contain the literal text, and quoted
# reply text (e.g. an email-style ``> see issue #123`` blockquote).
_FENCED_CODE_BLOCK_RE = re.compile(r"```.*?```", flags=re.DOTALL)
_BLOCKQUOTE_LINE_RE = re.compile(r"^[ \t]*>.*$", flags=re.MULTILINE)


def issue_numbers_mentioned_by_pr(pr: dict[str, Any]) -> set[int]:
    """Return issue numbers loosely referenced by a PR's title/body — advisory only.

    Matches the literal phrase ``issue #N`` / ``issues #N`` (case-insensitive,
    with or without a space between the word and the hash), after stripping
    fenced code blocks and blockquoted lines. This is a strict subset of
    GitHub's issue-reference syntax: it does not treat a bare ``#N`` (which
    could be a PR number) as an issue reference, and it does not treat
    closing keywords like ``Fixes #N`` as any more than a reference.

    This is looser than ``linked_issue_number``'s hijack-safety guarantee —
    phrases like "unlike issue #N", "follow-up to issue #N", or a collision
    with another repo's issue #N in the same text all still match, and there
    is no reliable lexical way to rule those out. Callers MUST treat a match
    as advisory only: it may be used to flag an issue for human review or
    exclude it from automation, but it must NEVER by itself authorize closing
    an issue or any other lifecycle-mutating action. Only ``linked_issue_number``
    (same-repo branch-prefix or closing-action verb) may authorize that.
    """
    text = f"{pr.get('title', '')}\n{pr.get('body', '')}"
    text = _FENCED_CODE_BLOCK_RE.sub("", text)
    text = _BLOCKQUOTE_LINE_RE.sub("", text)
    return {int(m.group(1)) for m in _ISSUE_MENTION_RE.finditer(text)}


def _is_not_found_gh_error(error: str) -> bool:
    """Classify a gh stderr/stdout string as an object-does-not-exist failure.

    Matches GitHub's GraphQL could-not-resolve shape and REST 404s — the same
    signals `_is_transient_gh_error` already treats as terminal. Permanent:
    retrying can never succeed while the referenced object is absent.
    """
    text = error.lower()
    if "could not resolve to a" in text or "not_found" in text:
        return True
    return bool(re.search(r"\bhttp 404\b", text))


def is_transient_repo_resolution_failure(error: str) -> bool:
    """Classify a ``GitHubNotFoundError`` message as a transient repository-level
    resolution failure rather than a permanent issue-level 404.

    GitHub's GraphQL emits distinct "Could not resolve to a X" messages:
    ``Could not resolve to a Repository with the name 'owner/repo'`` is a
    repository-level resolution failure, while ``Could not resolve to a Issue
    with the number N`` is an issue-level 404. Both match
    ``_is_not_found_gh_error``'s broad "could not resolve to a" pattern, so
    both raise ``GitHubNotFoundError`` — but only the issue-level 404 is
    permanent. A repository-level failure is transient: the orchestrator
    already successfully listed PRs from this repo (``pr_list`` at loop start),
    so the repo *did* resolve moments ago. The failure is a network/infra dip
    (issue #1132: a ~7-minute connectivity window produced exactly this shape
    and parked a PR as ``foreign_issue_ref`` for 32 hours).

    Returns True for repository-level resolution failures (transient); False
    for issue-level 404s and anything else (permanent or unknown).
    """
    return "could not resolve to a repository" in error.lower()


def _is_transient_gh_error(error: str) -> bool:
    """Classify a gh stderr/stdout string as a transient (retryable) failure.

    Transient signals are an allowlist; anything not explicitly listed is treated
    as terminal so genuine logic/auth/validation errors still fail fast.
    """
    text = error.lower()

    # Terminal signals that must never be retried.
    if "bad credentials" in text:
        return False
    if "could not resolve to a" in text or "not_found" in text:
        return False
    if re.search(r"\bhttp 401\b", text):
        return False
    # 403 is terminal unless it is a rate-limit/secondary-rate-limit response.
    if re.search(r"\bhttp 403\b", text) and not (
        "rate limit" in text
        or "secondary rate limit" in text
        or "was submitted too quickly" in text
    ):
        return False
    if re.search(r"\bhttp 422\b", text):
        return False

    # Transient allowlist.
    if "tls handshake timeout" in text:
        return True
    if "net/http:" in text:
        return True
    if "connection reset" in text:
        return True
    if "connection refused" in text:
        return True
    if "i/o timeout" in text:
        return True
    if re.search(r"\beof\b", text):
        return True
    if "timeout awaiting response headers" in text:
        return True
    if re.search(r"\bhttp (?:502|503|504|429)\b", text):
        return True
    if "was submitted too quickly" in text:
        return True
    if "you have exceeded a secondary rate limit" in text:
        return True
    # Primary and other GitHub rate-limit responses (often HTTP 403 or 429).
    if "rate limit" in text:
        return True
    if "error connecting to" in text:
        return True
    if "could not connect" in text:
        return True

    # Unknown errors are terminal by default.
    return False


def _is_pre_connection_error(error: str) -> bool:
    """Return True for failures that provably occurred before the request reached GitHub.

    Mutating commands are only retried on these pre-send errors to preserve
    at-most-once semantics; post-send ambiguous timeouts (i/o timeout, 5xx after
    headers, etc.) are surfaced immediately.
    """
    text = error.lower()
    return any(
        phrase in text
        for phrase in (
            "tls handshake timeout",
            "connection refused",
            "could not connect",
            "error connecting to",
        )
    )


def _should_retry(args: list[str], error: str, is_mutating: bool) -> bool:
    """Decide whether a failed gh invocation should be retried.

    Reads/idempotent commands may retry any transient failure. Mutating commands
    only retry provable pre-connection failures, avoiding double-application of
    merges, label edits, comments, etc.
    """
    if not _is_transient_gh_error(error):
        return False
    if not is_mutating:
        return True
    return _is_pre_connection_error(error)


def _graphql_field_value(args: list[str], field: str) -> str | None:
    """Return the raw value of a `gh api graphql -f/--field name=value` pair.

    Handles detached (`-f query=...`), attached shorthand (`-fquery=...`),
    and `--field=query=...` spellings (#919). Returns `None` if the field is
    absent or its value is missing.
    """
    for i, arg in enumerate(args):
        if arg in ("-f", "--raw-field", "-F", "--field"):
            next_arg = args[i + 1] if i + 1 < len(args) else ""
            if "=" in next_arg and next_arg.split("=", 1)[0] == field:
                return next_arg.split("=", 1)[1]
        elif arg.startswith("-f") and len(arg) > 2:
            rest = arg[2:].lstrip("=")
            if "=" in rest and rest.split("=", 1)[0] == field:
                return rest.split("=", 1)[1]
        elif arg.startswith(("--field=", "--raw-field=")):
            rest = arg.split("=", 1)[1]
            if "=" in rest and rest.split("=", 1)[0] == field:
                return rest.split("=", 1)[1]
    return None


def _is_graphql_query(args: list[str]) -> bool:
    """A `gh api graphql -f query='query { ... }'` is a read-only query.

    `args` is the argv after the leading `gh` token, so a GraphQL call looks
    like `["api", "graphql", "-f", "query=..."]`.

    Fails closed: only an operation that *starts* with the GraphQL `query`
    keyword is treated as read-only. `mutation` or anything unparseable is
    classified as mutating so a stray write never runs under `--dry-run`.
    """
    if len(args) < 2 or args[0] != "api" or args[1] != "graphql":
        return False
    query = _graphql_field_value(args, "query")
    if not query:
        return False
    return query.lstrip()[:5].lower() == "query"


def _api_is_mutating(args: list[str]) -> bool:
    """Classify a `gh api` invocation, for the --dry-run gate.

    `gh api` defaults to GET, so a bare `gh api <path>` is read-only and MUST stay
    runnable under --dry-run — roughly a dozen call sites (rate_limit, commits/{sha},
    check-runs, compare, branches/*/protection, `fleet_registry.py`) depend on that.
    Blanket-denying `api` would turn --dry-run from "observes without mutating" into
    "cannot observe", so the classification keys off whether a method is *named*.

    Structured on flag PRESENCE rather than on enumerating accepted spellings, and
    fails CLOSED when a method flag is present but its value cannot be extracted.
    The previous version enumerated `--method`/`--method=` only and fell through to
    False for everything else, so `-X DELETE` — the form `delete_branch` builds —
    classified as read-only and a --dry-run really deleted PR head branches
    (#914, #917). Enumeration is the wrong shape here: it silently fails open on
    each spelling nobody thought of (`-X`, `-X=`, `-XDELETE`, a trailing `--method`
    with no value).

    Read-only `gh api graphql -f query='query { ... }'` is an exception: it is a
    GraphQL query and must be runnable under `--dry-run` so `fleet status` can
    batch issue dependency/state lookups in a single subprocess (#923).
    """
    if _is_graphql_query(args):
        return False

    for i, arg in enumerate(args):
        if arg in ("-X", "--method"):
            method = args[i + 1] if i + 1 < len(args) else ""
        elif arg.startswith("--method="):
            method = arg.split("=", 1)[1]
        elif arg.startswith("-X"):
            # pflag shorthand accepts an attached value: `-XDELETE` and `-X=DELETE`.
            method = arg[2:].lstrip("=")
        else:
            continue
        # A named-but-unparseable method is not evidence of a read; fail closed.
        return not method or method.upper() not in ("GET", "HEAD")
    # No explicit method. gh switches GET -> POST when request parameters are added
    # ("adding request parameters will automatically switch the request method to
    # POST" -- gh api --help). Prefix-matched, not membership-tested, for the same
    # reason as the method arm: pflag accepts both the detached (`-f title=x`,
    # `--field=labels[]=bug`) and the attached (`-ftitle=x`) spelling, and a
    # membership test sees only the detached one (#919). `--field`/`--raw-field`/
    # `--input` are prefixes rather than exact matches so the bare and `=` forms
    # collapse into one condition.
    param_prefixes = ("--raw-field", "--field", "--input")
    return any(arg.startswith(param_prefixes) or arg[:2] in ("-f", "-F") for arg in args)


def _is_mutating(args: list[str]) -> bool:
    if not args:
        return False
    text = " ".join(args)
    # `gh api` defaults to GET and is read-only unless a mutating method is given.
    # run() passes args without the leading "gh" token.
    if text.startswith("api"):
        return _api_is_mutating(args)
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

# Markdown backtick code span: an opening run of backticks, content, and a
# closing run of the SAME length. Capturing group 2 is the span content.
_CODE_SPAN_RE = re.compile(r"(`+)(.+?)(\1)", flags=re.DOTALL)
# Balanced straight-double-quote span. Group 1 is the quoted content.
_DOUBLE_QUOTE_SPAN_RE = re.compile(r'"([^"]*)"')
# Opening fence of a fenced code block: a line beginning with a run of 3+
# backticks or tildes (optionally followed by an info string). CommonMark
# allows up to 3 leading spaces; we tolerate any leading whitespace.
_FENCE_OPEN_RE = re.compile(r"^[ \t]*([`~]{3,})")


def _inside_code_span(text: str, start: int, end: int) -> bool:
    """True if the [start, end) range falls inside a Markdown backtick code span."""
    for m in _CODE_SPAN_RE.finditer(text):
        if m.start(2) <= start and end <= m.end(2):
            return True
    return False


def _inside_quoted_span(text: str, start: int, end: int) -> bool:
    """True if the [start, end) range falls inside a straight-double-quote span."""
    for m in _DOUBLE_QUOTE_SPAN_RE.finditer(text):
        if m.start(1) <= start and end <= m.end(1):
            return True
    return False


def _fenced_block_ranges(text: str) -> list[tuple[int, int]]:
    """Return the ``(start, end)`` char-offset ranges of fenced code blocks.

    A fenced block starts with a line beginning with a run of 3+ backticks or
    tildes (optionally followed by an info string, e.g. ```` ```python ````)
    and ends at the next line beginning with a closing fence of the same
    character and at least the same length. An unclosed fence runs to the end
    of the text. Each returned range spans from the start of the opening fence
    line up to (excluding) the closing fence line, so any content line between
    the fences is contained in the range.
    """
    ranges: list[tuple[int, int]] = []
    pos = 0
    in_fence = False
    fence_char = ""
    fence_len = 0
    block_start = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip(" \t")
        if not in_fence:
            m = _FENCE_OPEN_RE.match(stripped)
            if m:
                fence_char = m.group(1)[0]
                fence_len = len(m.group(1))
                block_start = pos
                in_fence = True
        else:
            close_m = re.match(
                rf"{re.escape(fence_char)}{{{fence_len},}}[ \t]*$",
                stripped.rstrip("\r\n"),
            )
            if close_m:
                # Range covers opening fence line through last content line;
                # the closing fence line itself is excluded.
                ranges.append((block_start, pos))
                in_fence = False
                fence_char = ""
                fence_len = 0
        pos += len(line)
    if in_fence:
        # Unclosed fence runs to end of text.
        ranges.append((block_start, len(text)))
    return ranges


def _inside_fenced_block(text: str, start: int, end: int) -> bool:
    """True if the ``[start, end)`` range falls inside a fenced code block.

    Unlike inline code spans, fenced blocks are multi-line Markdown constructs
    whose opening and closing fence markers sit on separate lines from the
    content. Clause bounds (which break on newlines) therefore cannot detect
    them -- the content line is its own clause with no fence markers in it --
    so this check runs against the full document with absolute offsets, not
    the clause substring.
    """
    for r_start, r_end in _fenced_block_ranges(text):
        if r_start <= start and end <= r_end:
            return True
    return False


def _clause_bounds(text: str, match_start: int, match_end: int) -> tuple[int, int]:
    """Return the (start, end) offsets of the sentence/line containing a match.

    Bounded by the closest preceding AND following sentence terminator
    (".", "!", "?") or line break, so each bullet/sentence is judged
    independently. The boundary characters themselves are excluded from the
    returned range.
    """
    start_boundary = max(text.rfind(ch, 0, match_start) for ch in _CLAUSE_BOUNDARY_CHARS)
    start = start_boundary + 1 if start_boundary != -1 else 0
    end_candidates = [text.find(ch, match_end) for ch in _CLAUSE_BOUNDARY_CHARS]
    end_candidates = [p for p in end_candidates if p != -1]
    end = min(end_candidates) if end_candidates else len(text)
    return start, end


def is_infrastructure_failure(job: dict[str, Any], annotations: list[dict[str, Any]]) -> bool:
    """Detect if a failed job indicates infrastructure failure vs code failure.

    Returns True if the failed job shows signs of infrastructure failure:
    - Zero executed steps (billing lapse, runner never started)
    - Annotations matching "was not started" patterns (billing/runner issues)

    This is used to reclassify FAILURE-state checks as infra_failed instead of
    code failures, preventing rework worker dispatch against untested code.

    Issue #1383: the detection logic now lives in
    :func:`charlie_work.checks.is_infra_blocked_check`, which is config-driven
    (annotation patterns and the instant-fail threshold live in
    :class:`InfraBlockedConfig`, not hardcoded here). This function remains as
    a thin backward-compatible wrapper that delegates to the canonical
    classifier with a default config, so existing callers and tests keep
    working unchanged. New call sites should call
    ``is_infra_blocked_check`` directly with the active config.

    Args:
        job: A single job object with steps[] from the GitHub Actions API
        annotations: A flat list of annotation objects from the check-runs API

    Returns:
        True if any infrastructure failure signal is detected, False otherwise.
    """
    from .checks import is_infra_blocked_check
    from .config import InfraBlockedConfig

    return is_infra_blocked_check(job, annotations, InfraBlockedConfig())


def parse_blockers(text: str) -> list[int]:
    """Parse blocker issue numbers from issue body text.

    Returns a list of issue numbers declared as blockers using patterns like:
    - "Blocked by #N"
    - "Depends on #N"
    - "Blocked-by: #N"

    Handles comma-separated lists (e.g., "Blocked by #743, #744").

    A match is only treated as the CURRENT issue declaring its own blocker
    when it reads as a first-person declaration about THIS issue. Three guards
    enforce that, in order of structural strength:

    1. **Quoted/code exclusion** — a match falling inside a Markdown backtick
       code span, a triple-backtick (or ``~~~``) fenced code block, or a
       straight-double-quote span is prose quoting another issue's blocker
       declaration, not a self-declaration. This is the fix for issue #1454:
       an issue describing another issue's blocker phrase (backticked, quoted,
       or parenthetically annotated) must not self-gate. The inline span
       search is scoped to the containing clause (see guard 2) so an unrelated
       stray backtick or quote ELSEWHERE in the body cannot pair with a later
       one to envelope a genuine declaration and silently drop it -- a real
       false-negative risk in this backtick-heavy codebase. Fenced code blocks
       are multi-line constructs whose fence markers sit on separate lines
       from the content, so clause bounds (which break on newlines) cannot
       detect them; the fenced-block check therefore runs against the full
       document with absolute offsets, not the clause substring.
    2. **Foreign-issue-ref exclusion** — a match whose containing
       sentence/line carries ANY other ``#NNN`` reference (before OR after
       the match, and not part of the match itself) describes those OTHER
       issues, not this one. This generalizes the original backward-only
       ``_clause_preceding`` guard (issue #159) to also look forward, so
       issue-referencing parentheticals after the match are excluded too.
    3. The remaining matches are honored as genuine self-declarations.

    Returns an empty list if no blockers are found.
    """
    if not text:
        return []

    blockers: set[int] = set()
    # Check if they appear in blocker context
    for pattern in _BLOCKER_PATTERNS:
        for match in pattern.finditer(text):
            match_start, match_end = match.start(), match.end()

            # Guard 1a: a match inside a fenced code block (triple-backtick or
            # ~~~) is quoted prose, not a self-declaration. Fenced blocks are
            # multi-line constructs whose fence markers sit on separate lines
            # from the content, so the clause-scoped inline span checks below
            # cannot detect them (the content line is its own clause with no
            # fence markers). This check therefore runs against the full
            # document with absolute offsets (issue #1454 rework round 2).
            if _inside_fenced_block(text, match_start, match_end):
                continue

            # Both remaining guards judge the match against its containing
            # clause, so compute the clause window once and reuse it. Scoping
            # the inline span check to the clause is what prevents an unrelated
            # stray backtick/quote elsewhere in the body from swallowing a
            # genuine declaration (issue #1454 rework).
            clause_start, clause_end = _clause_bounds(text, match_start, match_end)
            clause = text[clause_start:clause_end]
            match_rel_start = match_start - clause_start
            match_rel_end = match_end - clause_start

            # Guard 1b: a match inside an inline code span or quoted span
            # WITHIN the clause is quoted prose describing another issue, not
            # a self-declaration. Searched on the clause substring so a span
            # opening outside this clause cannot envelope the match.
            if _inside_code_span(clause, match_rel_start, match_rel_end):
                continue
            if _inside_quoted_span(clause, match_rel_start, match_rel_end):
                continue

            # Guard 2: any OTHER #NNN in the containing clause (not part of
            # this match) means the clause is about a different issue.
            has_foreign_ref = False
            for ref in _ISSUE_REF.finditer(clause):
                if ref.start() >= match_rel_start and ref.end() <= match_rel_end:
                    continue  # part of the match itself
                has_foreign_ref = True
                break
            if has_foreign_ref:
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


def get_github_issue_dependencies(gh: GitHubLike, issue_number: int) -> list[int]:
    """Fetch GitHub's native issue dependencies (blocked_by relationships).

    Uses the GitHub API to check for issue dependencies. Tolerates 404/410 errors
    for repos that don't have the feature enabled. Returns an empty list on any
    error (fail-open for compatibility).

    Successful resolutions (a real dependency list, including a legitimate
    empty one, and the 404/410 "feature not available" case) are cached in
    ``gh``'s pass-scoped ``_list_cache`` keyed ``("issue_dependencies",
    issue_number)`` -- issue #870 found this call, made once per ready issue
    with zero caching, serial and uncached, was the single largest cost of
    `fleet status` (~140s of a ~184s run for 62 issues). Transient failures
    are deliberately NOT cached: caching a fail-open `[]` would silently
    erase a real dependency edge for the rest of this pass, which is a
    correctness change, not just a performance one -- so a transient error
    is retried on the next lookup within the same pass instead of being
    locked in.

    Args:
        gh: GitHub client instance. Duck-typed test doubles without a
            ``_list_cache`` attribute (several exist in tests/test_charlie_work.py,
            predating this cache) are tolerated -- caching is simply skipped
            for them rather than raising.
        issue_number: The issue number to check dependencies for

    Returns:
        List of issue numbers that block this issue via GitHub's native API
    """
    cache = getattr(gh, "_list_cache", None)
    cache_key = ("issue_dependencies", issue_number)
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

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
            # Transient error or gh not available — fail open with warning.
            # Not cached: see docstring.
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
        # Not cached: indistinguishable from a transient failure here.
        logger.warning(
            "GitHub dependencies API returned None for issue #%d - treating as no dependencies",
            issue_number,
        )
        return []
    elif isinstance(value, dict):
        # 404/410 error response — feature not available on this repo. This
        # is a stable, successful resolution (not a transient error), so it's
        # safe and correct to cache.
        if cache is not None:
            cache[cache_key] = []
        return []
    elif isinstance(value, list):
        # Extract issue numbers from the dependency list — a real, successful
        # resolution, cached.
        deps = [int(dep.get("number", 0)) for dep in value if dep.get("number")]
        if cache is not None:
            cache[cache_key] = deps
        return deps
    else:
        # Unexpected type — fail open with warning. Not cached.
        logger.warning(
            "GitHub dependencies API returned unexpected type %s for issue #%d - treating as no dependencies",
            type(value),
            issue_number,
        )
        return []


def cancel_superseded_runs(
    gh: GitHubLike,
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
