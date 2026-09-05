"""Issues capability: issue read/close and dependency surface (Track 2, #1585).

Cluster F of the design doc's capability segmentation (Section 3.1):
``close_issue``, ``issue_view``, ``issue_list``, ``issue_dependencies``,
``are_issues_open``.

Track 2, issue #1591; design doc Section 5, L07: all five Cluster F members
move here verbatim. ``are_issues_open`` and ``issue_view`` move together as a
unit (the must-not-split constraint): ``are_issues_open``'s thread-pool
fallback builds a closure that calls ``self.issue_view`` inside the pool, so
``issue_view`` must live on the same collaborator for that call to resolve on
``self`` (the ``Issues`` instance) rather than crossing a thread boundary back
through the owner delegate. ``issue_dependencies`` calls
``self._graphql_issue_dependencies``/``self._graphql_issue_states``, which stay
on the owner until L09 and resolve through ``CapabilityCollaborator.__getattr__``
either way, so they do NOT move in this leaf.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

# ``ci_fleet.github.GitHubError`` is imported directly here, not re-derived
# through ``charlie_work.github`` or ``_base.py``: ``close_issue``,
# ``are_issues_open`` and ``issue_dependencies`` (moved below) catch it, and
# identity matters -- ``github.py``'s own load-bearing comment on its
# ``GitHubError`` re-export (Track 2, issue #1585) explains why a local
# re-declaration would be a structurally-identical but *unrelated* exception
# type that no existing ``except GitHubError`` handler would catch.
# ``ci_fleet.github`` has no dependency on ``charlie_work``, so importing it
# directly here carries no circular-import risk (unlike ``charlie_work.github``
# itself, which imports ``github_capabilities`` before its own definitions are
# ready). Mirrors ``pull_requests.py``'s L06 precedent for the same reasoning.
from ci_fleet.github import GitHubError

# ``GitHubRunResult`` lives in ``_base.py``, not ``charlie_work.github`` (Track
# 2, issue #1588; design doc Section 5, L04) -- see ``_base.py`` for the full
# circular-import rationale. ``get_github_issue_dependencies`` (relocated below)
# performs a real runtime ``isinstance`` against it, so this is a normal
# top-level import, not a ``TYPE_CHECKING``-only one.
#
# ``_LIST_LIMIT`` also lives in ``_base.py`` (Track 2, issue #1590; design doc
# Section 5, L06) -- see ``_base.py``'s own comment for the cross-cutting
# rationale (it is shared with ``PullRequests`` members and, until this leaf,
# with ``GitHub.issue_list``). ``issue_list`` (moved below) references it as a
# bare global.
from ._base import CapabilityCollaborator, GitHubRunResult, _LIST_LIMIT

if TYPE_CHECKING:
    # Runtime import would cycle (github.py imports this module to build the
    # GitHubLike union); ``get_github_issue_dependencies``'s ``gh: GitHubLike``
    # annotation is a bare string under ``from __future__ import annotations``
    # and is never evaluated, so the TYPE_CHECKING-only import is all it needs.
    from charlie_work.github import GitHubLike

logger = logging.getLogger(__name__)

# Module-level constants for gh --json field lists, moved verbatim from
# ``github.py`` alongside ``issue_list``/``issue_view`` (Track 2, issue #1591;
# design doc Section 5, L07). ``issue_list``'s body references
# ``ISSUE_LIST_FIELDS`` and ``issue_view``'s body references
# ``ISSUE_VIEW_FIELDS`` as bare globals, so both must be bound in this module's
# globals. Re-exported through ``github_capabilities/__init__.py`` and
# re-imported into ``github.py`` (nothing there uses them directly anymore
# now that ``validate_field_lists`` moved to ``transport.py`` in L09, but
# ``doctor.py``/``test_github.py``/``test_doctor.py`` still read them via
# ``charlie_work.github.ISSUE_LIST_FIELDS``/``ISSUE_VIEW_FIELDS``) and
# directly into ``transport.py`` (Track 2, issue #1593; design doc Section 5,
# L09), which imports them from here rather than re-deriving a second copy --
# the same re-export pattern already used for
# ``PR_LIST_FIELDS``/``LABEL_LIST_FIELDS``.
ISSUE_LIST_FIELDS = "number,title,url,body,labels,author,createdAt,updatedAt,state"
ISSUE_VIEW_FIELDS = (
    "number,title,url,body,labels,assignees,author,comments,createdAt,updatedAt,state"
)

# Fan-out width for the per-issue ``issue_view`` fallback in
# ``are_issues_open`` (issue #870). Moved verbatim from ``github.py`` alongside
# ``are_issues_open`` (Track 2, issue #1591; design doc Section 5, L07). No
# other ``github.py`` consumer referenced this constant (only ``are_issues_open``
# did), so it is relocated *without* a re-export -- mirroring ``pull_requests.py``'s
# private ``_PR_URL_RE``.
#
# The network round trip dominates each per-issue state check (~1-7s observed),
# so this is a fan-out width, not a CPU budget -- picked to keep well clear of
# GitHub's secondary rate limits while still cutting a serial N x ~2s loop down
# substantially.
_MAX_ISSUE_STATE_WORKERS = 8


# Moved verbatim from ``github.py`` alongside ``issue_dependencies`` (Track 2,
# issue #1591; design doc Section 5, L07). ``issue_dependencies``'s fallback
# body calls it as a bare global, so it must live in this module's globals; it
# cannot be imported back from ``github.py`` (that path is circular -- github.py
# imports ``github_capabilities`` before its own definitions are ready). Its
# body touches only ``GitHubRunResult`` (imported from ``._base``), this
# module's ``logger``, and ``gh.*`` (duck-typed), so relocating it carries no
# circular-import risk. It is a *public* helper with external consumers --
# ``from charlie_work.github import get_github_issue_dependencies`` in
# ``backlog_reachability.py`` and ``workflow.py`` (plus ``test_backlog_reachability.py``
# and ``test_charlie_work.py``) -- so it is re-exported through
# ``github_capabilities/__init__.py`` and re-imported into ``github.py`` to keep
# that import path resolving to the same object. Body is unchanged -- do not
# edit the docstring or logic below, only this surrounding comment.
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


@runtime_checkable
class IssuesLike(Protocol):
    """Structural interface for issue read/close/dependency operations."""

    def close_issue(self, number: int) -> bool: ...

    def issue_view(self, number: int) -> dict[str, Any]: ...

    def issue_list(self, labels: Any = None, state: Any = None) -> list[dict[str, Any]]: ...

    def issue_dependencies(self, issue_numbers: list[int]) -> dict[int, list[int]]: ...

    def are_issues_open(self, issue_numbers: list[int]) -> set[int]: ...


class Issues(CapabilityCollaborator):
    """Issue read/close/dependency capability collaborator.

    Moved from ``GitHub`` verbatim (Track 2, issue #1591; design doc Section
    5, L07) -- all five Cluster F members. Bodies still say
    ``self.run(...)``/``self._list_cache``/``self._list_json(...)``, which
    resolve through ``CapabilityCollaborator.__getattr__`` to the owner (design
    doc Section 3.3).

    ``are_issues_open`` and ``issue_view`` move together (the must-not-split
    constraint): ``are_issues_open``'s thread-pool fallback closure calls
    ``self.issue_view``, which -- because ``issue_view`` is now a real lexical
    method on this collaborator -- resolves on ``self`` (the ``Issues``
    instance) rather than crossing the thread boundary back through the owner
    delegate. A class-level ``monkeypatch.setattr(GitHub, "issue_view", ...)``
    therefore no longer intercepts that specific internal call (it does still
    intercept external ``gh.issue_view(...)`` calls via the delegate); this
    interception-path relocation is disclosed in the L07 PR body.

    ``issue_dependencies`` calls ``self._graphql_issue_dependencies`` /
    ``self._graphql_issue_states`` (transport internals staying on the owner
    until L09) and, in its fallback, the module-level
    ``get_github_issue_dependencies`` relocated above. ``are_issues_open``
    likewise calls ``self._graphql_issue_states`` on the owner. Several bodies
    also reference module-level bare globals relocated alongside them:
    ``issue_list`` uses ``_LIST_LIMIT`` (from ``._base``) and
    ``ISSUE_LIST_FIELDS``; ``issue_view`` uses ``ISSUE_VIEW_FIELDS``;
    ``are_issues_open`` uses ``_MAX_ISSUE_STATE_WORKERS`` and
    ``ThreadPoolExecutor``; ``close_issue``/``are_issues_open``/
    ``issue_dependencies`` catch ``GitHubError`` (imported directly from
    ``ci_fleet.github`` -- see this module's import block). Design doc Section
    3.3 covers only ``self.<attr>`` forwarding, not bare-global runtime symbols
    in moved bodies; this is the same disclosed design-gap resolution that
    recurs identically in L04/L05/L06.
    """

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
