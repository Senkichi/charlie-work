from __future__ import annotations

import json
import logging
import random
import re
import subprocess
import time
from collections.abc import Callable, Iterable, Iterator
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

from .checks import _run_id_from_link  # noqa: F401  (deliberate re-export)
from .github_capabilities import (
    ChecksLike,
    CommentsLike,
    GitHubRunResult,
    ISSUE_LIST_FIELDS,  # noqa: F401  (deliberate re-export; doctor.py et al.)
    ISSUE_VIEW_FIELDS,  # noqa: F401  (deliberate re-export; doctor.py et al.)
    IssuesLike,
    LABEL_LIST_FIELDS,  # noqa: F401  (deliberate re-export; doctor.py et al.)
    LabelsLike,
    MERGED_PR_LIST_FIELDS,
    MergeBranchLike,
    PR_CHECKS_FIELDS,  # noqa: F401  (deliberate re-export; doctor.py et al.)
    PR_LIST_FIELDS,  # noqa: F401  (deliberate re-export; doctor.py et al.)
    PR_VIEW_FIELDS,  # noqa: F401  (deliberate re-export; doctor.py et al.)
    PullRequestsLike,
    RECONCILE_ISSUE_FIELDS,  # noqa: F401  (deliberate re-export; doctor.py)
    RECONCILE_PR_FIELDS,  # noqa: F401  (deliberate re-export; doctor.py)
    RUN_LIST_FIELDS,
    RepoMetaLike,
    _ADMIN_FLAG,
    _STRATEGY_FLAGS,
    _is_mutating,
)

# ``_LIST_LIMIT`` is no longer referenced inside ``github.py`` itself -- its
# last internal consumer, ``issue_list``, moved to the ``Issues`` collaborator
# in this leaf (Track 2, issue #1591; design doc Section 5, L07). It stays a
# deliberate re-export because ``reconcile.py`` (``from .github import
# _LIST_LIMIT``) and the test suite (``test_reconcile.py``/``test_charlie_work.py``
# via ``charlie_work.github._LIST_LIMIT``) still read it from here.
from .github_capabilities import _LIST_LIMIT  # noqa: F401  (deliberate re-export)
from .github_capabilities import _job_id_from_link  # noqa: F401  (deliberate re-export)
from .github_capabilities import _pr_number_from_url  # noqa: F401  (deliberate re-export)
from .github_capabilities import (  # noqa: F401  (deliberate re-export)
    get_github_issue_dependencies,
)
from .github_delegation import _COLLABORATORS, _install_delegates
from .github_delegation import _ROUTES, _SIGNATURE_SOURCE, _make_delegate  # noqa: F401 (deliberate re-export)
from .subprocess_runner import no_console_window_kwargs

logger = logging.getLogger(__name__)

# Conventional exit status for "killed by timeout" (GNU coreutils `timeout`).
# A TimeoutExpired carries no returncode of its own, and callers that branch on
# returncode must not see a 0 that reads as success.
_TIMEOUT_RETURNCODE = 124

# Fractional jitter applied to each retry backoff (e.g. 0.25 => +/- 25%).
_JITTER_FRACTION = 0.25

# _DEFAULT_GH_MAX_RETRIES/_DEFAULT_GH_RETRY_BASE_SECONDS/
# _DEFAULT_GH_TIMEOUT_SECONDS, _GRAPHQL_BATCH_SIZE/_GRAPHQL_BLOCKED_BY_FIRST,
# and _GIT_REMOTE_URL_RE/_parse_git_remote_url moved to
# github_capabilities/transport.py alongside _max_retries/_retry_base_seconds/
# _timeout_seconds, _graphql_issue_states/_graphql_issue_dependencies, and
# _repo_owner_name respectively (Track 2, issue #1593; design doc Section 5,
# L09) -- no consumer of any of them remains in this module, so none is
# re-exported.

# Module-level constants for gh --json field lists.
# These are the single source of truth for all JSON field queries to GitHub.
# All call sites must use these constants — no inline field-list literals.
PR_VIEW_MERGED_FIELDS = "state,mergedAt,headRefOid"
# MERGED_PR_LIST_FIELDS (the field contract for every merged-PR listing) moved
# to github_capabilities/_base.py (Track 2, issue #1593; design doc Section 5,
# L09), imported above -- it is a bare global in both merged_prs_for_issue()
# (stays on GitHub) and Transport.validate_field_lists() (moved this leaf).
# See _base.py's comment for the full field-contract rationale (unchanged).
#
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
# RECONCILE_PR_FIELDS/RECONCILE_ISSUE_FIELDS moved to
# github_capabilities/transport.py alongside validate_field_lists (Track 2,
# issue #1593; design doc Section 5, L09), imported above as a pure
# re-export -- nothing in this module uses them directly anymore, but
# doctor.py still reads them via `from .github import RECONCILE_PR_FIELDS`.
# RUN_LIST_FIELDS moved to github_capabilities/_base.py alongside
# MERGED_PR_LIST_FIELDS above (same leaf) -- still used directly below by
# cancel_superseded_runs.

# Flag constants for merge_pr — single source of truth for both argv construction
# and config validation. Derive ORCHESTRATOR_MANAGED_MERGE_FLAGS from these so that
# adding a new orchestrator-managed flag to merge_pr automatically rejects it in
# config validation (prevents drift issue #107).
#
# _STRATEGY_FLAGS/_ADMIN_FLAG moved to github_capabilities/merge_branch.py
# alongside merge_pr (Track 2, issue #1592; design doc Section 5, L08) --
# merge_pr's body references both as bare globals, so they must be bound in
# that module's globals. Re-exported through github_capabilities/__init__.py
# and re-imported below because ORCHESTRATOR_MANAGED_MERGE_FLAGS (this
# module-level constant, not a GitHub member, so it stays here) also needs
# them -- the same disclosed design-gap resolution (design doc Section 3.3
# covers only self.<attr> forwarding, not bare-global runtime symbols) that
# recurs identically across leaves.
_DELETE_BRANCH_FLAG = "--delete-branch"
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


@dataclass(frozen=True)
class GitHub:
    repo_root: Path
    dry_run: bool = False
    runtime: RuntimeConfig | None = None

    # _max_retries/_retry_base_seconds/_timeout_seconds moved to
    # github_capabilities/transport.py (Track 2, issue #1593; design doc
    # Section 5, L09) -- reached from here (e.g. by `run` below) through the
    # installed `_transport` delegate.

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

    # _normalize_rest_pr moved to github_capabilities/transport.py (Track 2,
    # issue #1593; design doc Section 5, L09) -- reached through the
    # installed `_transport` delegate.

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

    # _run_bool/_list_json moved to github_capabilities/transport.py (Track 2,
    # issue #1593; design doc Section 5, L09) -- reached through the
    # installed `_transport` delegate.

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

    # _pr_checks_fallback/validate_field_lists/_repo_owner_name/
    # _graphql_query/_graphql_issue_states/_graphql_issue_dependencies moved
    # to github_capabilities/transport.py (Track 2, issue #1593; design doc
    # Section 5, L09) -- reached through the installed `_transport` delegate.


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
