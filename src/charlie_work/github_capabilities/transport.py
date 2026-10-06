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
``_max_retries``, ``_retry_base_seconds``,
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

from ._base import CapabilityCollaborator, GitHubRunResult
from ._field_probes import field_list_probes, probe_verdict
from .circuit_breaker_transport import (
    circuit_breaker_open_message,
    circuit_breaker_state_path,
    note_circuit_breaker_result,
)
from .cross_repo_blockers import CrossRepoBlocker, make_blocker
from ._send import read_json, send, send_graphql
from .graphql_issue_states import graphql_issue_states
from ..github_transport.json_read import JsonRead
from ..github_transport.request import GraphQLRequest, RestRequest
from ..subprocess_runner import no_console_window_kwargs

logger = logging.getLogger(__name__)

# Defaults used when GitHub is constructed without a RuntimeConfig (tests and
# legacy callers). Production code should pass config.runtime so these are
# configurable via orchestrator.config.yaml. Moved from ``github.py`` verbatim
# alongside ``_max_retries``/``_retry_base_seconds``/``_timeout_seconds``
# (Track 2, issue #1593; design doc Section 5, L09). No other consumer
# referenced these three, so they are relocated without a re-export.
#
# Lowered from 120.0 to 30.0 (issue #1833, follow-up to the #1832 overnight
# outage): a connect/handshake-class failure or a genuine hang should fail
# fast, not tie up a serial retry loop for two minutes per attempt. Calls
# with a legitimately long response body (large paginated lists) opt into
# ``_DEFAULT_GH_LONG_CALL_TIMEOUT_SECONDS`` via ``run(..., long_call=True)``
# instead of raising this shared default.
_DEFAULT_GH_MAX_RETRIES = 3
_DEFAULT_GH_RETRY_BASE_SECONDS = 1.0
_DEFAULT_GH_TIMEOUT_SECONDS = 30.0

# Budget for calls that are known to legitimately take longer than the
# fail-fast default above (issue #1833) -- large paginated list/search
# responses, not a hang. ``_list_json`` (every ``issue list``/``pr list`` call)
# and ``merged_pr_list``'s manual REST pagination loop are the only two call
# shapes that opt in; see their ``long_call=True`` call sites.
_DEFAULT_GH_LONG_CALL_TIMEOUT_SECONDS = 120.0

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

# Field list for the ``statusCheckRollup`` probe ``validate_field_lists`` runs
# (issue #1609). ``_pr_checks_fallback``, its original caller, was deleted
# with the ``gh pr checks`` dependency (ADR-0006, B7).
PR_STATUS_CHECK_ROLLUP_FIELDS = "statusCheckRollup"


class Transport(CapabilityCollaborator):
    """Shared low-level transport capability collaborator.

    Twelve members moved verbatim from ``GitHub`` (Track 2, issue #1593;
    design doc Section 5, L09): ``_max_retries``, ``_retry_base_seconds``,
    ``_timeout_seconds``, ``_normalize_rest_pr``, ``_run_bool``,
    ``_list_json``, ``validate_field_lists``,
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

    def _long_call_timeout_seconds(self) -> float:
        """Budget for a call known in advance to be legitimately long-running
        (large paginated list/search responses -- see ``_list_json`` and
        ``merged_pr_list``'s ``long_call=True`` call sites), as opposed to
        ``_timeout_seconds``'s fail-fast default for everything else
        (issue #1833).
        """
        if self.runtime is not None:
            return self.runtime.gh_long_call_timeout_seconds
        return _DEFAULT_GH_LONG_CALL_TIMEOUT_SECONDS

    def _circuit_breaker_allow_call(self) -> bool:
        """Whether ``GitHub.run()`` may spawn ``gh`` right now (issue #1833).

        Delegates to the owner's single persistent ``CircuitBreakerState``
        (constructed once in ``GitHub.__post_init__`` via
        ``circuit_breaker_transport.build_circuit_breaker_state``, alongside
        ``_list_cache`` -- both are mutable per-instance runtime state, not
        config). ``False`` means: the caller must fail this invocation as a
        value without running a subprocess, matching the errors-as-values
        invariant.
        """
        return self._circuit_breaker_state.allow_call()

    def _circuit_breaker_open_message(self, command: list[str]) -> str:
        """Thin wrapper: this is a ``self.<name>()`` delegation target
        reached from ``GitHub.run()``, so it must stay declared directly on
        ``Transport``'s own class body (``github_delegation._build_routes()``
        only routes members from a collaborator class's own ``__dict__``).
        The message-building logic itself lives in
        ``circuit_breaker_transport.circuit_breaker_open_message`` (issue
        #1833 follow-up, file-size ratchet issue #1442).
        """
        return circuit_breaker_open_message(self._circuit_breaker_state, command)

    def _circuit_breaker_note_result(
        self, *, error: str | None = None, timed_out: bool = False
    ) -> None:
        """Thin wrapper, same reason as ``_circuit_breaker_open_message``
        above: a ``self.<name>()`` delegation target from ``GitHub.run()``,
        so it stays on ``Transport``. The classification/recording/event
        logic lives in
        ``circuit_breaker_transport.note_circuit_breaker_result``.
        """
        note_circuit_breaker_result(
            self._circuit_breaker_state,
            circuit_breaker_state_path(self.runtime, self.repo_root),
            error=error,
            timed_out=timed_out,
        )

    def reset_circuit_breaker(self) -> None:
        """Per-pass reset hook (issue #1833).

        Called from ``OrchestratorApp._loop_body`` alongside
        ``invalidate_list_cache()``, for the same reason: a long-running
        ``charlie fleet supervise`` process reuses one ``GitHub`` instance
        across many passes, so per-pass state must be explicitly re-armed at
        the top of each pass rather than living for the life of the process --
        otherwise a breaker tripped by one bad pass's network blip would
        permanently fail-fast every later pass.
        """
        self._circuit_breaker_state.reset()

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
            # Issue #1803: the merge timestamp, mapped to gh's GraphQL name.
            # The mention gate's temporal rule compares this against the
            # issue's createdAt -- a PR merged before the issue existed
            # cannot have addressed it. REST spells it `merged_at`; the
            # normalized shape must carry it forward or every REST-sourced
            # merged PR reads as timestamp-unknown downstream.
            "mergedAt": pr.get("merged_at"),
        }

    def _run_bool(self, args: list[str]) -> bool:
        """Run a gh command and return True iff returncode == 0.

        This is a private helper for label operations that need boolean success
        semantics without inferring from stdout/stderr string shape. Never raises
        — failures are returned as False (allow_failure semantics). Dry-run mode
        returns True (the operation would succeed if not for dry-run).
        """
        result = self.run(args, allow_failure=True)
        if not isinstance(result, GitHubRunResult):
            return True  # dry-run: ``run`` answered a mutating argv with a bare string
        return result.ok

    def _list_json(self, read: JsonRead, *, kind: str) -> list[dict[str, Any]]:
        # The guarded transport applies the fleet-wide bounded retry policy,
        # so no ad-hoc retry loop here. Callers build the read with
        # long_call=True (issue #1833): a list of hundreds of items
        # legitimately takes longer than the fail-fast default.
        limit = read.limit
        result = read_json(self, read)
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

    def validate_field_lists(self) -> None:
        """Validate the compile-time field lists against the live GitHub schema.

        Sends one ``first:1`` / ``number:0`` probe per registered list through
        the guarded transport (B13: GraphQL schema validation replaces gh's
        ``Available fields`` stderr). GitHub rejects an unknown field before it
        executes anything, so a ``undefinedField`` error names exactly the
        configured field the schema lacks; a NOT_FOUND for ``number:0`` proves
        the selection was accepted. Raises ``ConfigError`` naming the constant
        and the offending field(s) ONLY on that positive rejection. Anything
        else (a transport-class failure, 401/403/404, primary or secondary rate
        limit, an inconclusive probe) is not a config error: warn and skip, so
        startup never depends on GitHub availability or quota (issue #1833).
        """
        # Import lazily to avoid the config -> github import cycle.
        from ..config import ConfigError

        try:
            owner, name = self._repo_owner_name()
        except GitHubError as exc:
            logger.warning("Skipping gh field-list validation this pass: %s", exc)
            return

        for constant, fields, probe in field_list_probes(
            RECONCILE_PR_FIELDS, RECONCILE_ISSUE_FIELDS
        ):
            outcome = (
                send(self, probe)
                if isinstance(probe, RestRequest)
                else probe.execute(self._transport_v2, owner, name)
            )
            verdict = probe_verdict(outcome)
            if verdict.skip:
                logger.warning(
                    "Could not validate field list %s due to a transport-class failure; "
                    "skipping remaining field-list validation this pass: %s",
                    constant,
                    verdict.detail,
                )
                return
            if verdict.rejected is not None:
                missing = [f for f in fields.split(",") if f in verdict.rejected] or list(
                    verdict.rejected
                )
                raise ConfigError(
                    f"GitHub does not support field(s) for {constant}: {', '.join(missing)}"
                )
            if verdict.detail:
                # Inconclusive (neither accepted nor rejected): not evidence of a bad
                # field list, so startup must not depend on it. Warn and move on.
                logger.warning(
                    "Could not validate field list %s (inconclusive probe); skipping it: %s",
                    constant,
                    verdict.detail,
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

    def _graphql_query(self, query: str) -> dict[str, Any]:
        """Run a single read-only GraphQL query via ``gh api graphql``.

        A query is a read, so ``--dry-run`` does not suppress it. Raises GitHubError for non-zero exit or a response that
        contains no usable ``data``.
        """
        owner, name = self._repo_owner_name()
        value, error = send_graphql(self, GraphQLRequest.of(query, {"owner": owner, "name": name}))
        if error is not None:
            raise GitHubError(f"GraphQL query failed: {error}")

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

        Returns a mapping ``issue_number -> is_open`` covering the requested
        numbers that resolved. A number absent from the mapping failed to
        resolve inside the batch (a per-node ``Could not resolve`` error):
        ``are_issues_open`` per-issue-fetches exactly those numbers instead
        of demoting the whole batch (issue #1933). Whole-query failures still
        raise ``GitHubError``.

        The implementation lives in ``graphql_issue_states.py`` as a free
        function -- the same split ``circuit_breaker_transport.py`` made for
        this module's other helpers (file-size ratchet #1442, attachment-point
        ceiling on ``Transport``).
        """
        return graphql_issue_states(self, issue_numbers, _GRAPHQL_BATCH_SIZE)

    def _graphql_issue_dependencies(self, issue_numbers: list[int]) -> dict[int, list[int]]:
        """Fetch GitHub-native ``blockedBy`` dependencies for many issues at once.

        Returns a mapping ``issue_number -> [blocker, ...]`` where a blocker is
        a plain ``int`` (same repo) or a ``CrossRepoBlocker`` (issue #2005).
        Also warms the ``("issue_open", blocker_number)`` cache -- or
        ``("issue_open", repo, blocker_number)`` for a cross-repo blocker --
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
                f"nodes {{ number state repository {{ nameWithOwner }} }} "
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
                                # Issue #2005: a blocker in another repo keeps
                                # its repo identity and is cached under a
                                # repo-qualified key, never the bare number.
                                repo_info = node.get("repository")
                                node_repo = (
                                    repo_info.get("nameWithOwner")
                                    if isinstance(repo_info, dict)
                                    else None
                                )
                                blocker = make_blocker(
                                    int(blocker_number), node_repo, f"{owner}/{name}"
                                )
                                blocked_by.append(blocker)
                                # Cache the blocker state now so
                                # are_issues_open does not need to re-derive it.
                                state = str(node.get("state") or "").upper()
                                if isinstance(blocker, CrossRepoBlocker):
                                    if state in ("OPEN", "CLOSED"):
                                        self._list_cache[
                                            ("issue_open", blocker.repo, int(blocker))
                                        ] = state == "OPEN"
                                else:
                                    self._list_cache[("issue_open", int(blocker_number))] = (
                                        state == "OPEN"
                                    )

                deps_by_number[number] = blocked_by
                self._list_cache[("issue_dependencies", number)] = blocked_by

        return deps_by_number
