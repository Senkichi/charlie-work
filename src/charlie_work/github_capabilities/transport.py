"""Transport capability: shared low-level GitHub CLI/HTTP plumbing (#1585).

Not a ``GitHubLike`` sub-protocol cluster. Transport is the destination for
the non-protocol internals (design doc Section 3.2: ``_run_bool``,
``_list_json``, ``_graphql_query``, the retry-knob helpers, etc.) that are
call targets from every other cluster but are not themselves part of the
public ``GitHubLike`` surface. ``run`` and ``__post_init__`` stay on the
owner (``GitHub``) as the interception seam and dataclass hook respectively
-- they never move here.

Track 2, issue #1593; design doc Section 5, L09 (the final leaf): moves the
twelve members below verbatim -- ``_run_bool``, ``_list_json``,
``_repo_owner_name``, ``_graphql_query``, ``_graphql_issue_states``,
``_graphql_issue_dependencies``, ``_normalize_rest_pr``,
``_pr_checks_fallback``, ``_max_retries``, ``_retry_base_seconds``,
``_timeout_seconds``, ``validate_field_lists``. ``_max_retries``/
``_retry_base_seconds``/``_timeout_seconds`` carry no property or other
decorator (design doc Section 3.1's decorator invariant) -- confirmed by
reading their source directly -- so no property-shaped delegate extension to
``github_delegation.py`` is needed.

``validate_field_lists``'s body contains one necessary, mechanical deviation
from byte-identical verbatim: its local ``from .config import ConfigError``
resolves relative to the *containing module's* package. In ``github.py``
(package ``charlie_work``) that is ``charlie_work.config``; copied unchanged
into this module (package ``charlie_work.github_capabilities``) the same text
would resolve to the nonexistent ``charlie_work.github_capabilities.config``.
Fixed by bumping the import to ``from ..config import ConfigError``, which
resolves to the identical ``charlie_work.config.ConfigError`` object -- the
AST differs only in ``ImportFrom.level`` (1 -> 2), never in the runtime
target. The import stays local/lazy (not hoisted to module level) because the
existing comment's cycle warning is real: ``config`` -> ``github`` ->
``github_capabilities`` -> ``transport`` would cycle if this module imported
``charlie_work.config`` at import time.
"""

from __future__ import annotations

import logging
import re
import subprocess
from typing import Any

from ci_fleet.github import GitHubError

from ._base import (
    CapabilityCollaborator,
    GitHubRunResult,
    MERGED_PR_LIST_FIELDS,
    RUN_LIST_FIELDS,
    _is_mutating,
)
from .checks import PR_CHECKS_FIELDS
from .issues import ISSUE_LIST_FIELDS, ISSUE_VIEW_FIELDS
from .labels import LABEL_LIST_FIELDS
from .pull_requests import PR_LIST_FIELDS, PR_VIEW_FIELDS
from ..subprocess_runner import no_console_window_kwargs

logger = logging.getLogger(__name__)

# Defaults used when GitHub is constructed without a RuntimeConfig (tests and
# legacy callers). Production code should pass config.runtime so these are
# configurable via orchestrator.config.yaml. Moved from ``github.py`` verbatim
# alongside ``_max_retries``/``_retry_base_seconds``/``_timeout_seconds``
# (Track 2, issue #1593; design doc Section 5, L09). No other consumer
# referenced these three, so they are relocated without a re-export.
_DEFAULT_GH_MAX_RETRIES = 3
_DEFAULT_GH_RETRY_BASE_SECONDS = 1.0
_DEFAULT_GH_TIMEOUT_SECONDS = 120.0

# How many issue numbers to pack into one batched `gh api graphql` query.
# Kept conservative to stay under the ~32KB Windows command-line limit and
# GitHub's GraphQL node/complexity budgets. See issue #923. Moved from
# ``github.py`` verbatim alongside ``_graphql_issue_states``/
# ``_graphql_issue_dependencies`` (Track 2, issue #1593; design doc Section 5,
# L09). No other consumer referenced it, so it is relocated without a
# re-export.
_GRAPHQL_BATCH_SIZE = 50

# GitHub allows up to 50 blocked-by / blocking relationships per issue.
# `first:` counts nodes toward the query's complexity, so matching the product
# limit keeps the query cheap and avoids false negatives. Moved from
# ``github.py`` verbatim alongside ``_graphql_issue_dependencies`` (Track 2,
# issue #1593; design doc Section 5, L09). No other consumer referenced it, so
# it is relocated without a re-export.
_GRAPHQL_BLOCKED_BY_FIRST = 50

# Parse "owner/repo" out of common git remote URL shapes. Intentionally loose:
# it matches the tail `.../owner/repo(.git)?` of https/ssh/git URLs, including
# `https://token@host/owner/repo.git` and `git@github.com:owner/repo.git`.
# Moved from ``github.py`` verbatim alongside ``_parse_git_remote_url``/
# ``_repo_owner_name`` (Track 2, issue #1593; design doc Section 5, L09). No
# other consumer referenced it, so it is relocated without a re-export.
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


# Minimal field lists for drift detection (reconcile.py). headRefOid is a
# plain scalar (like state/title) -- NOT a per-item graph walk like
# statusCheckRollup (see the PR_CHECKS_FIELDS note in checks.py and issue
# #361); safe to include unconditionally. Needed by
# detect_aviator_stale_blocked's commit_check_runs(sha) lookup. Moved from
# ``github.py`` alongside ``validate_field_lists`` (Track 2, issue #1593;
# design doc Section 5, L09) -- referenced there as a bare global. Re-exported
# through ``github_capabilities/__init__.py`` and re-imported into
# ``github.py`` (nothing in ``github.py`` itself uses it directly anymore;
# kept as a pure re-export because ``doctor.py`` reads it via
# ``from .github import RECONCILE_PR_FIELDS``).
RECONCILE_PR_FIELDS = "number,title,url,headRefName,baseRefName,body,state,labels,isCrossRepository,headRefOid,closedAt"
RECONCILE_ISSUE_FIELDS = "number,title,url,body,labels,state"


class Transport(CapabilityCollaborator):
    """Shared low-level transport capability collaborator.

    Twelve members moved verbatim from ``GitHub`` (Track 2, issue #1593;
    design doc Section 5, L09): ``_max_retries``, ``_retry_base_seconds``,
    ``_timeout_seconds``, ``_normalize_rest_pr``, ``_run_bool``,
    ``_list_json``, ``_pr_checks_fallback``, ``validate_field_lists``,
    ``_repo_owner_name``, ``_graphql_query``, ``_graphql_issue_states``,
    ``_graphql_issue_dependencies``. ``run`` and ``__post_init__`` stay on the
    owner permanently (module docstring above).

    Several of these call each other via ``self.<name>`` and, because all
    twelve move together in this one leaf, those calls now resolve directly
    on this class rather than crossing back through
    ``CapabilityCollaborator.__getattr__``: ``_run_bool``/``_list_json`` call
    ``self.run`` (stays on the owner -- unaffected); ``_graphql_query`` calls
    ``self._repo_owner_name``; ``_graphql_issue_states``/
    ``_graphql_issue_dependencies`` call both ``self._repo_owner_name`` and
    ``self._graphql_query``; ``validate_field_lists`` calls
    ``self._timeout_seconds``. An exhaustive census (every ``GitHub.<name>``
    attribute access, ``patch``/``monkeypatch.setattr`` at class, instance,
    and ``patch.object`` granularity, and every ``FakeGitHub``-lineage double)
    found zero existing tests patch any of these specific internal-call
    targets in a way this same-collaborator resolution would bypass -- unlike
    L07's ``are_issues_open``/``issue_view`` pair, there is no positive
    bypass instance to fix here, only the same hazard *class* to document
    (disclosed in the L09 PR body with a positive control demonstrating what
    a bypass would look like).

    Bodies still say ``self.run(...)``/``self.dry_run``/``self.repo_root``/
    ``self.runtime``/``self._list_cache``, which resolve through
    ``CapabilityCollaborator.__getattr__`` to the owner (design doc Section
    3.3). Several also reference module-level bare globals relocated
    alongside them: ``_max_retries``/``_retry_base_seconds``/
    ``_timeout_seconds`` use the ``_DEFAULT_GH_*`` constants above;
    ``_graphql_issue_states``/``_graphql_issue_dependencies`` use
    ``_GRAPHQL_BATCH_SIZE``/``_GRAPHQL_BLOCKED_BY_FIRST``; ``_repo_owner_name``
    uses ``_parse_git_remote_url``/``_GIT_REMOTE_URL_RE``;
    ``validate_field_lists`` uses ten field-list constants (three defined
    above, seven imported from the other capability modules that already own
    them) and ``ConfigError`` (see the module docstring's disclosed import-depth
    fix). Design doc Section 3.3 covers only ``self.<attr>`` forwarding, not
    bare-global runtime symbols in moved bodies; this is the same disclosed
    design-gap resolution that recurs identically across every leaf.
    """

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
            # this merge -- the only post-merge anchor from which "was this
            # content already on main?" can still be answered, since after
            # the merge everything the PR carried is main-reachable through
            # the merge commit itself.
            "mergeCommitOid": pr.get("merge_commit_sha"),
        }

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

    def _pr_checks_fallback(self, number: int) -> list[dict[str, Any]] | None:
        """Disambiguate a ``gh pr checks`` failure via ``statusCheckRollup`` (issue #846).

        ``gh pr checks`` cannot represent "no checks reported yet" as a
        successful empty response -- it exits non-zero with empty stdout for
        that case, identically to a genuine command failure (see the caller).
        ``gh pr view --json statusCheckRollup`` CAN: it returns a clean empty
        list with exit 0 for a PR with zero checks (measured against PR #700
        in this repo: ``{"statusCheckRollup":[]}``, exit 0).

        This field carries its own risk -- it is a per-item GraphQL graph walk
        that can fail on token scope (see the PR_CHECKS_FIELDS note in
        github_capabilities/checks.py and issue #361) -- but any failure here
        simply falls through to this function's ``None`` return, which is
        exactly pr_checks' pre-existing
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
          PR_CHECKS_FIELDS note in github_capabilities/checks.py). Every
          consumer that reads "bucket"
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

    def validate_field_lists(self) -> None:
        """Validate the compile-time ``--json`` field lists against ``gh``.

        Probes each list with an invalid field and parses the ``Available
        fields:`` section of the stderr. Fails fast with a ``ConfigError`` naming
        the constant and the offending field(s) when the installed ``gh`` CLI
        does not support a configured field.
        """
        # Import lazily to avoid the config -> github import cycle.
        from ..config import ConfigError

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
