"""GitHub CI-check reads: pr_checks, check-run annotations, field-list validation.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

from pathlib import Path
import pytest
from _fake_transport import (
    FakeAdapter,
    check_run,
    checks_reply,
    connection_page,
    failure,
    graphql_failure,
    graphql_variables,
    make_github,
    ok,
    sent,
)
from charlie_work import github as github_module
from charlie_work.github_transport.outcome import FailureKind, GraphQLError, Response
from charlie_work.github_transport.request import RestRequest
from charlie_work.config import (
    ConfigError,
    RuntimeConfig,
)


def _schema_accepting_reply(request):
    """What GitHub answers when every probed selection is valid: an empty
    connection for a list, NOT_FOUND for the ``number: 0`` views, empty REST."""
    if isinstance(request, RestRequest):
        return ok({"workflow_runs": []} if "actions/runs" in request.route else [])
    if "number" in graphql_variables(request):
        return graphql_failure("Could not resolve to a node with the number of 0.")
    connection = "issues" if "issues(" in request.document else "pullRequests"
    return connection_page(connection, [])


def test_pr_checks_fields_excludes_database_id() -> None:
    """Regression guard: gh pr checks --json does not support "databaseId".

    Adding it to PR_CHECKS_FIELDS (unlike gh run list --json, which does
    support it) makes the installed gh CLI exit non-zero with 'Unknown JSON
    field: "databaseId"'. Because pr_checks() uses allow_failure=True and
    treats a non-list result as "no checks", this silently returns [] from
    EVERY pr_checks() call — summarize_checks() then reports all required
    checks "missing" and merge_ready() computes can_merge=False for every PR,
    killing the entire auto-merge lane. This exact string broke the merge lane
    on 2026-07-10. The job id workflow.py needs is instead derived from "link"
    by pr_checks() via _job_id_from_link().
    """
    fields = github_module.PR_CHECKS_FIELDS.split(",")
    assert "databaseId" not in fields
    assert "link" in fields


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        (
            "https://github.com/OWNER/REPO/actions/runs/123456/job/789012",
            789012,
        ),
        (
            "https://github.com/OWNER/REPO/actions/runs/123456/job/789012/",
            789012,
        ),
        (
            "https://github.com/OWNER/REPO/actions/runs/123456/job/789012?check_suite_focus=true",
            789012,
        ),
        (
            "https://github.com/OWNER/REPO/actions/runs/123456/job/789012#step:3:1",
            789012,
        ),
        ("https://example.com/some/external/status-check", None),
        ("", None),
        (None, None),
    ],
)
def test_job_id_from_link(link, expected) -> None:
    assert github_module._job_id_from_link(link) == expected


def test_pr_checks_injects_database_id_from_link(tmp_path: Path) -> None:
    """pr_checks() derives databaseId from link for Actions checks, None otherwise."""
    reply = checks_reply(
        check_run("Tests passed", url="https://github.com/OWNER/REPO/actions/runs/1/job/42"),
        {
            "__typename": "StatusContext",
            "context": "external-status-check",
            "state": "SUCCESS",
            "targetUrl": "https://example.com/status",
        },
    )
    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))

    checks = gh.pr_checks(123)

    assert {c["name"]: c["databaseId"] for c in checks} == {
        "Tests passed": 42,
        "external-status-check": None,
    }


def test_pr_checks_returns_empty_list_on_empty_success(tmp_path: Path) -> None:
    """A PR with no check contexts returns [], not None."""
    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", [checks_reply()]))

    assert gh.pr_checks(123) == []


def test_pr_checks_returns_none_on_gh_command_failure(tmp_path: Path) -> None:
    """A read the API rejected (a GraphQL error) returns None."""
    reply = graphql_failure("Field 'bogus' doesn't exist on type 'CheckRun'", "undefinedField")
    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))

    assert gh.pr_checks(123) is None


def test_check_run_annotations_returns_parsed_list_on_success(tmp_path: Path) -> None:
    annotations = [{"path": "src/foo.py", "start_line": 42, "message": "line too long"}]
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok(annotations)]))

    result = gh.check_run_annotations(999)

    assert result == annotations
    assert sent(http) == [("GET", "repos/{owner}/{repo}/check-runs/999/annotations", None)]


def test_check_run_annotations_returns_empty_list_on_api_failure(tmp_path: Path) -> None:
    """Issue #771: the annotations accessor must return a value (empty list),
    never raise, when the API call fails -- callers building required_changes
    from it must never crash the review() codepath on a transient GitHub error."""
    gh, _, _ = make_github(
        tmp_path, http=FakeAdapter("http", [ok({"message": "Not Found"}, status=404)])
    )

    assert gh.check_run_annotations(999) == []


def test_pr_checks_returns_list_when_checks_fail(tmp_path: Path) -> None:
    """A failing check is a normal row (bucket "fail"), not a command failure."""
    reply = checks_reply(check_run("Tests", "FAILURE"))
    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))

    checks = gh.pr_checks(123)

    assert checks == [
        {
            "name": "Tests",
            "state": "FAILURE",
            "bucket": "fail",
            "link": "",
            "databaseId": None,
            "runId": None,
        }
    ]


def test_validate_field_lists_passes_when_gh_lists_all_fields(tmp_path: Path) -> None:
    """Startup self-check accepts field lists the GitHub schema supports (B13)."""
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", handler=_schema_accepting_reply))

    gh.validate_field_lists()

    # Should have probed all 10 field-list constants, one request each.
    assert len(http.api_requests) == 10


def test_validate_field_lists_fails_on_unsupported_field(tmp_path: Path) -> None:
    """Startup self-check fails fast if a configured field is not in the schema."""
    reply = Response(
        200,
        (),
        '{"data": null}',
        "http",
        graphql_errors=(
            GraphQLError("Field 'nonexistent' doesn't exist on type 'Issue'", "undefinedField"),
        ),
    )
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))

    with pytest.raises(ConfigError, match="ISSUE_LIST_FIELDS.*nonexistent"):
        gh.validate_field_lists()
    assert len(http.api_requests) == 1


def test_validate_field_lists_timeout_raises_config_error(tmp_path: Path) -> None:
    """A hung probe during startup field-list validation is
    transport-class, not a configuration error (issue #1833, follow-up to the
    #1832 outage): it must no longer raise ConfigError -- doing so propagated
    uncaught out of OrchestratorApp.__init__ via fleet_dispatch.py's
    "Error processing repo" path, violating the errors-as-values invariant.
    validate_field_lists() now returns normally, skipping the remaining
    probes for this pass, and the guard records the failure on the circuit
    breaker. The probe is bound by the configured gh_timeout_seconds."""
    timed_out = failure(FailureKind.TIMEOUT, "hung")
    http = FakeAdapter("http", [timed_out])
    gh, http, _ = make_github(
        tmp_path,
        http=http,
        gh=FakeAdapter("gh", [failure(FailureKind.TIMEOUT, "hung", "gh")], token="tok-1"),
        runtime=RuntimeConfig(gh_timeout_seconds=45.0, gh_max_retries=0),
    )

    gh.validate_field_lists()  # must not raise

    # Fails fast on the first field list probed, not after trying all 10.
    assert len(http.api_requests) == 1
    # The timeout the adapter was given came from the configured
    # gh_timeout_seconds, not a hardcoded value.
    assert [call.timeout for call in http.calls if call.request in http.api_requests] == [45.0]
    # The transport-class failure is recorded on the breaker.
    assert gh._transport._circuit_breaker_state.consecutive_failures == 1
    assert gh._transport.__dict__.get("_list_cache") is None
