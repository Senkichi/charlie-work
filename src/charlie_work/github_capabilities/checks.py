"""Checks capability: CI check/run inspection (Track 2, issue #1585).

Cluster C of the design doc's capability segmentation (Section 3.1):
``pr_checks``, ``check_run_annotations``, ``commit_check_runs``,
``actions_job``, ``workflow_runs_for_head``, ``check_graphql_rate_limit``.

``check_graphql_rate_limit`` is an ambiguity call (Section 3.1): it wraps a
GraphQL rate probe used by the checks path, though it is transport-adjacent.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from ..checks import _run_id_from_link
from ._base import CapabilityCollaborator, GitHubRunResult

# Moved from ``github.py`` alongside ``check_graphql_rate_limit`` (Track 2,
# issue #1588; design doc Section 5, L04). No other ``github.py`` consumer
# referenced this default (only the moved method's own default-arg value),
# so it is relocated without a re-export -- unlike ``PR_CHECKS_FIELDS``
# below.
_DEFAULT_GRAPHQL_RATE_LIMIT_THRESHOLD = 1500

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
#
# Moved from ``github.py`` alongside ``pr_checks`` (Track 2, issue #1588;
# design doc Section 5, L04). ``pr_checks``'s body is byte-identical to its
# former ``GitHub`` copy and still references this constant as a bare global
# name, so it must be bound in *this* module's globals (see the rationale in
# ``labels.py``'s ``LABEL_LIST_FIELDS`` comment). Re-exported through
# ``github_capabilities/__init__.py`` and re-imported into ``github.py``
# (nothing there uses it directly anymore now that ``validate_field_lists``
# moved to ``transport.py`` in L09, but it is also read by ``doctor.py`` and
# tests via ``charlie_work.github.PR_CHECKS_FIELDS``) and directly into
# ``transport.py`` (Track 2, issue #1593; design doc Section 5, L09), which
# imports it from here rather than re-deriving a second copy -- the same
# re-export pattern already used there for ``GitHubError`` and
# ``LABEL_LIST_FIELDS``.
PR_CHECKS_FIELDS = "name,state,bucket,link"

# Matches the job-id segment of a GitHub Actions check link, e.g.
# https://github.com/OWNER/REPO/actions/runs/RUN_ID/job/JOB_ID (optionally
# followed by a query string or #fragment, e.g. "?check_suite_focus=true").
#
# Moved from ``github.py`` alongside ``_job_id_from_link``/``pr_checks``
# (Track 2, issue #1588; design doc Section 5, L04). No other ``github.py``
# consumer referenced this regex, so it is relocated without a re-export.
_ACTIONS_JOB_LINK_RE = re.compile(r"/actions/runs/\d+/job/(\d+)")


# Moved from ``github.py`` verbatim alongside ``pr_checks`` (Track 2, issue
# #1588; design doc Section 5, L04). Its body is unchanged -- do not edit the
# docstring or logic below, only this surrounding comment -- because
# ``tests/_fakes_github.py`` imports it directly (``from charlie_work.github
# import _job_id_from_link, ...``) and ``tests/test_charlie_work.py`` reads
# it via ``github_module._job_id_from_link``. Re-exported through
# ``github_capabilities/__init__.py`` and re-imported into ``github.py``,
# same as ``PR_CHECKS_FIELDS`` above.
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


@runtime_checkable
class ChecksLike(Protocol):
    """Structural interface for CI check/run inspection operations."""

    def pr_checks(self, number: int) -> list[dict[str, Any]] | None: ...

    def check_run_annotations(self, check_run_id: int) -> list[dict[str, Any]]: ...

    def commit_check_runs(self, sha: str) -> list[dict[str, Any]] | None: ...

    def actions_job(self, job_id: int) -> dict[str, Any] | None: ...

    def workflow_runs_for_head(self, head_sha: str) -> list[dict[str, Any]] | None: ...

    def check_graphql_rate_limit(self, threshold: int = ...) -> tuple[bool, int, int | None]: ...


class Checks(CapabilityCollaborator):
    """CI check/run inspection capability collaborator.

    Moved from ``GitHub`` verbatim (Track 2, issue #1588; design doc Section
    5, L04). Bodies still say ``self.run(...)``, which resolves through
    ``CapabilityCollaborator.__getattr__`` to the owner's ``run`` (design doc
    Section 3.3). ``pr_checks`` additionally calls
    ``self._pr_checks_fallback(...)`` -- still a lexical ``GitHub`` method
    until L09 (Transport) -- which resolves the same way; order L04-before-L09
    is safe either way.

    Two of the six also reference module-level bare globals relocated
    alongside them: ``pr_checks`` uses ``PR_CHECKS_FIELDS``,
    ``_job_id_from_link``, and ``_run_id_from_link`` (the last still defined
    in the top-level ``charlie_work.checks`` module); all six use
    ``GitHubRunResult``, relocated to ``_base.py`` and re-exported through
    ``github.py`` rather than imported back from it, to avoid a circular
    import (``github.py`` imports ``github_capabilities`` before its own,
    now-former, ``GitHubRunResult`` definition -- see ``_base.py`` for the
    full rationale). Design doc Section 3.3 covers only ``self.<attr>``
    forwarding, not bare-global runtime symbols in moved bodies; this is a
    disclosed design-gap resolution that recurs identically in L05/L06/L08.
    """

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
