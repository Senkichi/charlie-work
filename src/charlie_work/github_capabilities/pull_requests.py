"""Pull-requests capability: PR read/create surface (Track 2, issue #1585).

Cluster E of the design doc's capability segmentation (Section 3.1):
``pr_create``, ``pr_view``, ``pr_list``, ``pr_diff``, ``pr_commits``,
``pr_ready``, ``merged_pr_list``, ``merged_prs_for_issue``.

``merged_prs_for_issue``/``merged_pr_list`` are an ambiguity call
(Section 3.1): pinned here because they return PR data even though
``merged_prs_for_issue`` is issue-keyed.

Track 2, issue #1590; design doc Section 5, L06: seven of Cluster E's eight
members move here (``pr_create``, ``pr_list``, ``merged_pr_list``,
``pr_view``, ``pr_diff``, ``pr_commits``, ``pr_ready``). ``merged_prs_for_issue``
does NOT move in this leaf -- its verbatim body calls ``linked_issue_number(...)``
as a bare global, a ~70-line, non-``GitHub``-method utility in ``github.py``
with dozens of external call sites across ``janitor.py``, ``reconcile.py``,
``dead_worker_reap.py``, ``cli.py``, ``backlog_reachability.py``,
``worktree.py``, and ``workflow.py`` (17+ sites alone), plus its own
dependency chain (one member of which, ``iter_unnegated_closing_keyword_matches``,
has its own external consumer in ``closing_keyword_gate``). Relocating that
whole surface would be a second, much larger Mikado leaf of its own, not a
same-leaf bare-global relocation like ``GitHubRunResult``/``_is_mutating``/
``_LIST_LIMIT`` below -- disclosed as a deviation from issue #1590's 8-member
list in this leaf's PR body, with a follow-up leaf recommended rather than
fixed inline here. ``merged_prs_for_issue`` therefore stays a lexical
``GitHub`` method; ``PullRequestsLike`` still declares it (protocol
conformance is satisfied by the unmoved lexical method on ``GitHub``, and by
``CapabilityCollaborator.__getattr__`` forwarding to it on the
``PullRequests`` collaborator side), and ``MergedPRSearchResult`` stays a
``TYPE_CHECKING``-only import here -- used only in that protocol stub's
return annotation, which ``inspect.signature`` never evaluates
(``eval_str=False`` by default).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

# ``ci_fleet.github.GitHubError`` is imported directly here, not re-derived
# through ``charlie_work.github`` or ``_base.py``: ``merged_pr_list`` (moved
# below) raises it, and identity matters -- ``github.py``'s own load-bearing
# comment on its ``GitHubError`` re-export (Track 2, issue #1585) explains why
# a local re-declaration would be a structurally-identical but *unrelated*
# exception type that no existing ``except GitHubError`` handler would catch.
# ``ci_fleet.github`` has no dependency on ``charlie_work``, so importing it
# directly here carries no circular-import risk (unlike ``charlie_work.github``
# itself, which imports ``github_capabilities`` before its own definitions are
# ready). Mirrors ``repo_meta.py``'s L05 precedent for the same reasoning.
from ci_fleet.github import GitHubError

# ``GitHubRunResult`` lives in ``_base.py``, not ``charlie_work.github`` (Track
# 2, issue #1588; design doc Section 5, L04) -- see ``_base.py`` for the full
# circular-import rationale. Three of the seven members moved below
# (``pr_diff``, ``pr_commits``, ``pr_ready``) perform a real runtime
# ``isinstance``/construction use of it, so (unlike before L06) this must be a
# normal top-level import, not the ``TYPE_CHECKING``-only one this module used
# while none of its own members had moved yet -- mirroring ``repo_meta.py``'s
# L05 promotion of the same import for the same reason.
#
# ``_LIST_LIMIT``/``_is_mutating`` also live in ``_base.py`` (Track 2, issue
# #1590; design doc Section 5, L06) -- see ``_base.py``'s own comment on each
# for the full cross-cutting rationale (both are shared with ``GitHub``
# methods that have not moved yet, so they belong in the shared base, not
# here).
from ._base import CapabilityCollaborator, GitHubRunResult, _is_mutating, _LIST_LIMIT

if TYPE_CHECKING:
    # Runtime import would cycle (github.py imports this module to build the
    # GitHubLike union); inspect.signature does not evaluate annotations
    # (eval_str=False by default), so the bare string name is all the
    # conformance test's signature comparison needs. Stays TYPE_CHECKING-only
    # even after L06 (unlike GitHubRunResult above) because the real
    # ``MergedPRSearchResult(...)`` construction happens inside
    # ``merged_prs_for_issue``, which this leaf deliberately does not move
    # (see the module docstring).
    from charlie_work.github import MergedPRSearchResult

logger = logging.getLogger(__name__)

# Moved from ``github.py`` verbatim alongside ``pr_list``/``pr_view`` (Track 2,
# issue #1590; design doc Section 5, L06). ``pr_list``'s body still references
# ``PR_LIST_FIELDS`` as a bare global name, and ``pr_view``'s default argument
# value binds ``PR_VIEW_FIELDS`` at *def* time, so both must be bound in this
# module's globals before the class body below executes (see the rationale in
# ``labels.py``'s ``LABEL_LIST_FIELDS`` comment). Re-exported through
# ``github_capabilities/__init__.py`` and re-imported into ``github.py``
# (nothing there uses them directly anymore now that ``validate_field_lists``
# moved to ``transport.py`` in L09, but ``doctor.py``/``test_github.py``/
# ``test_janitor.py`` still read them via
# ``charlie_work.github.PR_LIST_FIELDS``/``PR_VIEW_FIELDS``) and directly into
# ``transport.py`` (Track 2, issue #1593; design doc Section 5, L09), which
# imports them from here rather than re-deriving a second copy -- the same
# re-export pattern already used for ``PR_CHECKS_FIELDS``/``LABEL_LIST_FIELDS``.
PR_LIST_FIELDS = "number,title,url,headRefName,baseRefName,body,isDraft,labels,author,updatedAt,reviewDecision,statusCheckRollup,headRefOid,isCrossRepository,mergeStateStatus,mergeable,state"
PR_VIEW_FIELDS = "number,title,url,headRefName,baseRefName,body,isDraft,labels,author,updatedAt,reviewDecision,statusCheckRollup,state,mergeable,additions,deletions,headRefOid,isCrossRepository,mergeStateStatus"

# Matches the PR-number segment of a pull-request URL, e.g.
# https://github.com/OWNER/REPO/pull/123
#
# Moved from ``github.py`` alongside ``_pr_number_from_url``/``pr_create``
# (Track 2, issue #1590; design doc Section 5, L06). No other ``github.py``
# consumer referenced this regex, so it is relocated without a re-export --
# mirroring ``checks.py``'s ``_ACTIONS_JOB_LINK_RE``.
_PR_URL_RE = re.compile(r"/pull/(\d+)")


# Moved from ``github.py`` verbatim alongside ``pr_create`` (Track 2, issue
# #1590; design doc Section 5, L06). Its body is unchanged -- do not edit the
# docstring or logic below, only this surrounding comment -- because
# ``tests/test_github_pr_create.py`` imports it directly (``from
# charlie_work.github import GitHub, _pr_number_from_url``) and exercises it
# with a dedicated parametrized test. Re-exported through
# ``github_capabilities/__init__.py`` and re-imported into ``github.py``, same
# as ``PR_LIST_FIELDS``/``PR_VIEW_FIELDS`` above -- mirroring ``checks.py``'s
# ``_job_id_from_link``.
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


@runtime_checkable
class PullRequestsLike(Protocol):
    """Structural interface for pull-request read/create operations."""

    def pr_create(self, head: str, base: str, title: str, body: str) -> int | None: ...

    def pr_view(self, number: int, *, fields: str = ...) -> dict[str, Any]: ...

    def pr_list(self) -> list[dict[str, Any]]: ...

    def pr_diff(self, number: int) -> str: ...

    def pr_commits(self, number: int) -> list[dict[str, Any]] | None: ...

    def pr_ready(self, number: int) -> GitHubRunResult: ...

    def merged_pr_list(self) -> list[dict[str, Any]]: ...

    def merged_prs_for_issue(self, issue_number: int, branch_prefix: str) -> MergedPRSearchResult:
        """Return merged PRs binding to ``issue_number``.

        The returned object is list-like and carries an ``ok`` flag. Callers
        must check ``ok`` before treating an empty result as "no merged PRs";
        ``ok=False`` means the search itself failed (rate limit, etc.).
        """
        ...


class PullRequests(CapabilityCollaborator):
    """Pull-request read/create capability collaborator.

    Moved from ``GitHub`` verbatim (Track 2, issue #1590; design doc Section
    5, L06) -- seven of Cluster E's eight members; see the module docstring
    for why ``merged_prs_for_issue`` stays on ``GitHub``. Bodies still say
    ``self.run(...)``/``self._list_cache``/``self._normalize_rest_pr(...)``/
    ``self.dry_run``, which resolve through ``CapabilityCollaborator.__getattr__``
    to the owner (design doc Section 3.3).

    Several members also reference module-level bare globals relocated
    alongside them: ``pr_list``/``pr_view`` use ``PR_LIST_FIELDS``/
    ``PR_VIEW_FIELDS`` (the latter also as ``pr_view``'s default argument
    value, bound at *def* time); ``pr_create`` uses ``_pr_number_from_url``
    and this module's own ``logger``; ``pr_diff``/``pr_commits``/``pr_ready``
    use ``GitHubRunResult`` (relocated to ``_base.py`` in L04 and imported
    from there, not re-derived from ``github.py``, to avoid a circular
    import); ``pr_list``/``merged_pr_list`` use ``_LIST_LIMIT`` and
    ``pr_ready`` uses ``_is_mutating`` (both relocated to ``_base.py`` in this
    leaf because they are also used by ``GitHub`` methods that have not moved
    yet -- see ``_base.py``'s own comments on each); and ``merged_pr_list``
    raises ``GitHubError`` (imported directly from ``ci_fleet.github``, the
    same external, identity-sensitive source ``github.py`` itself re-exports
    from -- see this module's import block). Design doc Section 3.3 covers
    only ``self.<attr>`` forwarding, not bare-global runtime symbols in moved
    bodies; this is the same disclosed design-gap resolution that recurs
    identically in L04/L05/L06/L08/L09.
    """

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
