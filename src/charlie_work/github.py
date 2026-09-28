from __future__ import annotations

import json
import logging
import random
import re
import subprocess
import time
from collections.abc import Callable, Iterable
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
from .github_body_scan import (  # noqa: F401  (deliberate re-export)
    detect_prose_only_dependencies,
    issue_numbers_mentioned_by_pr,
    parse_blockers,
)
from .github_capabilities import (
    ChecksLike,
    CommentsLike,
    GitHubRunResult,
    build_circuit_breaker_state,
    build_http_transport_state,
    ISSUE_LIST_FIELDS,  # noqa: F401  (deliberate re-export; doctor.py et al.)
    ISSUE_VIEW_FIELDS,  # noqa: F401  (deliberate re-export; doctor.py et al.)
    IssuesLike,
    LABEL_LIST_FIELDS,  # noqa: F401  (deliberate re-export; doctor.py et al.)
    LabelsLike,
    MERGED_PR_LIST_FIELDS,  # noqa: F401  (deliberate re-export; test_github.py et al.)
    MergeBranchLike,
    MergedPRSearchResult,
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
    run_gh_command,
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

# ``_CLOSING_KEYWORDS_ALT`` is imported from ``issue_linking.py`` (Track 2,
# issue #1613; design doc Section 5, L06b) as a real (not re-export-only)
# name: ``_CLOSING_KEYWORD_DEFANG_RE`` below still references it as a bare
# global and stays in this module (only the ``#N``-matching half of the
# vocabulary moved to ``issue_linking``; the defang/rewrite half did not, per
# issue #1613's own scope). ``linked_issue_number``,
# ``iter_unnegated_closing_keyword_matches`` and ``_CLOSING_KEYWORD_REF`` used
# to be re-exported here too, but every importer (``workflow.py``,
# ``reconcile.py``, ``janitor.py``, ``cli.py``, ``dead_worker_reap.py``,
# ``backlog_reachability.py``, ``worktree.py``, ``closing_keyword_gate.py``,
# ``closing_reference.py``, plus tests) now imports them directly from
# ``issue_linking`` (issue #1627), so the re-exports are gone -- the names are
# no longer reachable through ``charlie_work.github``.
from .issue_linking import _CLOSING_KEYWORDS_ALT
from .transient_errors import is_transient_network_error

logger = logging.getLogger(__name__)

# Conventional exit status for "killed by timeout" (GNU coreutils `timeout`).
# A TimeoutExpired carries no returncode of its own, and callers that branch on
# returncode must not see a 0 that reads as success.
_TIMEOUT_RETURNCODE = 124

# Sentinel returncode for "the circuit breaker refused this call" (issue
# #1833) -- no gh subprocess ever ran, so there is no real exit status.
# Distinct from _TIMEOUT_RETURNCODE: a caller branching on returncode needs
# to tell "gh hung" from "gh was never spawned" apart.
_CIRCUIT_OPEN_RETURNCODE = 125

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
# Field list for the worktree-GC fallback PR lookup (issue #1713): when
# state.json carries no linked PR for a dispatch-prefixed worktree branch,
# ``clean_worktrees`` runs ``gh pr list --head <branch> --state all`` and
# needs only the PR number -- the resolved number feeds the existing live
# ``gh pr view`` confirmation above, which re-fetches merge state rather
# than trusting the list row.
WORKTREE_PR_HEAD_FIELDS = "number"
# MERGED_PR_LIST_FIELDS (the field contract for every merged-PR listing) moved
# to github_capabilities/pull_requests.py (Track 2, issue #1613; design doc
# Section 5, L06b), imported above -- it is a bare global in both
# merged_prs_for_issue() (moved there too, this leaf) and
# Transport.validate_field_lists() (moved to transport.py in L09; imports the
# constant from pull_requests.py). See pull_requests.py's comment for the
# full field-contract rationale (unchanged).
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
# field by field. Issue #1872 added `baseRefName`/`headRefOid`: the gate
# re-resolves the PR's merge base against the LIVE base ref (compare API)
# because the recorded `base.sha` can lag behind main after a sync merge,
# false-positiving foreign commits into the scanned commit surface. Both are
# scalar fields -- no nested connection, so no statusCheckRollup-style scope
# risk.
CLOSING_KEYWORD_PR_FIELDS = "title,body,headRefName,baseRefName,headRefOid,isCrossRepository"
# Fields for the post-create closing-reference verification (cw#1263): the
# only field needed is GitHub's own GraphQL resolution of which issues this
# PR will close on merge -- as opposed to `linked_issue_number`'s regex-based
# guess, `closingIssuesReferences` is GitHub's authoritative answer. Kept as
# narrow as `CLOSING_KEYWORD_PR_FIELDS` for the same reason: no CI/review
# state is needed, so no `statusCheckRollup` token-scope risk.
PR_CLOSING_ISSUES_FIELDS = "closingIssuesReferences"
# The single ``number`` field used by doctor.py's probe helpers
# (``_find_pr_number``/``_find_issue_number``) to discover a real item number
# for the live field-list validation pass. Kept as a constant rather than an
# inline literal so the field-list lint (tests/test_doctor.py::
# test_gh_field_lists_use_constants_no_inline_literals) covers the
# single-positional-list ``gh.run([...], json_output=True)`` call shape too
# (issue #1609). Named generically because the same field list serves both the
# PR and issue probes.
PROBE_NUMBER_FIELDS = "number"
# RECONCILE_PR_FIELDS/RECONCILE_ISSUE_FIELDS moved to
# github_capabilities/transport.py alongside validate_field_lists (Track 2,
# issue #1593; design doc Section 5, L09), imported above as a pure
# re-export -- nothing in this module uses them directly anymore, but
# doctor.py still reads them via `from .github import RECONCILE_PR_FIELDS`.
# RUN_LIST_FIELDS moved to github_capabilities/_base.py (Track 2, issue
# #1593; design doc Section 5, L09) -- still used directly below by
# cancel_superseded_runs. (MERGED_PR_LIST_FIELDS, once alongside it there,
# moved on to github_capabilities/pull_requests.py in L06b -- see the comment
# above.)

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


# MergedPRSearchResult moved to github_capabilities/pull_requests.py alongside
# merged_prs_for_issue (Track 2, issue #1613; design doc Section 5, L06b),
# imported above -- kept as a plain re-export because tests/_fakes_github.py,
# tests/_reconcile_fixtures.py, tests/_salvage_fixtures.py, and
# tests/test_charlie_work.py all construct it via
# ``charlie_work.github.MergedPRSearchResult``/``._MergedPRSearchResult``.
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
        # Per-pass circuit breaker state (issue #1833): mutable, per-instance
        # runtime state constructed once here, exactly like _list_cache above
        # -- see reset_circuit_breaker() (reached through the _transport
        # delegate below) for why it must be explicitly re-armed every pass
        # rather than living for the process lifetime. Built here (not
        # lazily) because build_circuit_breaker_state needs self.runtime,
        # which is already available at this point, and because it must be
        # constructed before the _COLLABORATORS loop below installs the
        # delegate that exposes it.
        object.__setattr__(
            self, "_circuit_breaker_state", build_circuit_breaker_state(self.runtime)
        )
        # Per-instance pooled HTTP transport state (issue #1834): mutable,
        # constructed once here for the same reason as
        # `_circuit_breaker_state` above -- `run_gh_command` needs it before
        # the `_COLLABORATORS` loop below installs any delegate.
        object.__setattr__(self, "_http_transport_state", build_http_transport_state())
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
        self,
        args: list[str],
        *,
        json_output: bool = False,
        allow_failure: bool = False,
        long_call: bool = False,
    ) -> Any:
        command = ["gh", *args]
        if self.dry_run and _is_mutating(args):
            return [] if json_output else "DRY-RUN: " + " ".join(command)

        # Per-pass circuit breaker gate (issue #1833): after enough
        # consecutive transport-class failures this pass, fail every further
        # call immediately as a value -- never spawn gh -- until the cooldown
        # elapses. Checked before the retry loop, not inside it: an open
        # breaker must skip the subprocess entirely, not merely skip retries
        # on one.
        if not self._circuit_breaker_allow_call():
            breaker_error = self._circuit_breaker_open_message(command)
            if not allow_failure:
                raise GitHubError(breaker_error)
            return GitHubRunResult(
                ok=False,
                returncode=_CIRCUIT_OPEN_RETURNCODE,
                stdout="",
                stderr=breaker_error,
                value=None,
                error=breaker_error,
            )

        is_mutating = _is_mutating(args)
        max_retries = self._max_retries()
        base_delay = self._retry_base_seconds()
        timeout_seconds = (
            self._long_call_timeout_seconds() if long_call else self._timeout_seconds()
        )
        last_result: subprocess.CompletedProcess[str] | None = None

        for attempt in range(max_retries + 1):
            try:
                # Issue #1834: `run_gh_command` chooses HTTP or the `gh`
                # subprocess per call (config-default HTTP for
                # `http_translate.is_http_candidate` shapes, `gh` for
                # everything else or on per-call HTTP fallback) and always
                # returns a `subprocess.CompletedProcess`-shaped result (or
                # raises the same two exceptions a direct `subprocess.run`
                # call would) -- every line below this point is unchanged
                # regardless of which transport actually produced `result`.
                result = run_gh_command(
                    args=args,
                    command=command,
                    cwd=self.repo_root,
                    timeout_seconds=timeout_seconds,
                    runtime=self.runtime,
                    transport_state=self._http_transport_state,
                    resolve_owner_repo=self._repo_owner_name,
                )
            except FileNotFoundError as exc:
                # Transport-class by construction (issue #1833): gh could not
                # even be spawned, so there is no response to classify --
                # reuse timed_out=True, which classify_gh_failure() always
                # treats as transport regardless of message text.
                self._circuit_breaker_note_result(timed_out=True)
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
                    # Terminal for this run() call -- record now, not on the
                    # retryable branch below, since the breaker observes
                    # each call's FINAL outcome, not every intra-call attempt
                    # (issue #1833).
                    self._circuit_breaker_note_result(timed_out=True)
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
                # A response reached the caller -- the transport is provably
                # fine right now regardless of what gh's exit code says
                # elsewhere, so record it before branching (issue #1833).
                self._circuit_breaker_note_result()
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
        # Terminal for this run() call: classify and record now (issue
        # #1833). A response DID reach the caller here (unlike the
        # TimeoutExpired/FileNotFoundError branches above), so this goes
        # through classify_gh_failure() on the actual stderr text rather than
        # a forced transport classification.
        self._circuit_breaker_note_result(error=final_error)
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

    # merged_prs_for_issue moved to github_capabilities/pull_requests.py
    # (Track 2, issue #1613; design doc Section 5, L06b) -- reached through
    # the installed `_pull_requests` delegate. Its body's only sibling call
    # is `self.run(...)`; `run` is never itself a routed/collaborator-side
    # member (it always resolves via `CapabilityCollaborator.__getattr__` to
    # this owner), so no subclass-override bypass hazard applies here (unlike
    # L07's `are_issues_open`/`issue_view`).

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
        self,
        args: list[str],
        *,
        json_output: bool = False,
        allow_failure: bool = False,
        long_call: bool = False,
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


# _CLOSING_KEYWORDS_ALT (imported above from issue_linking.py) is GitHub's own
# issue-closing keyword set. _CLOSING_KEYWORD_REF/_BRANCH_ISSUE_REF/the
# negation-lookback constants and helpers/iter_unnegated_closing_keyword_matches/
# _first_unnegated_closing_keyword_match/linked_issue_number all moved to
# issue_linking.py alongside it (Track 2, issue #1613; design doc Section 5,
# L06b) -- only _CLOSING_KEYWORD_DEFANG_RE below (the defang/rewrite half of
# the vocabulary, not the matching half) stays in this module.
# Rewrites `<keyword> #N` to `<keyword> issue N` — used by
# `defang_closing_keywords` to strip the auto-close/binding syntax from text
# that will be embedded in a PR body/comment charlie-work does not control
# downstream (e.g. a rework brief a worker reads and copies into its own PR).
_CLOSING_KEYWORD_DEFANG_RE = re.compile(
    r"(" + _CLOSING_KEYWORDS_ALT + r")(\s+)#(\d+)", flags=re.IGNORECASE
)


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


# linked_issue_number moved to issue_linking.py alongside the rest of the
# closing-keyword chain (Track 2, issue #1613; design doc Section 5, L06b).
# It is no longer re-exported through this module: every importer was
# repointed to ``from .issue_linking import linked_issue_number`` directly
# (issue #1627), so the name is not reachable through
# ``charlie_work.github``.


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


# _is_transient_gh_error's allowlist moved to transient_errors.py (issue
# #1773): raw `git` network calls (git_retry.py) hit the identical failure
# class over the identical network path and need the identical
# classification, so this is now a thin alias onto the shared definition
# rather than a second, independently-maintained copy of the allowlist. See
# that module's docstring for the full rationale, including why the two
# git-specific patterns it adds ("could not resolve host", "connectex") are
# inert for gh's own Go-idiom error text and so do not change this
# function's behavior for any error gh can actually produce.
_is_transient_gh_error = is_transient_network_error


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
